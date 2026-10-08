from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import networkx as nx
import numpy as np
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from src.utils.configs import SNAPMSConfig


@dataclass
class DBMatch:
    neutral_mass: float
    smiles: str
    inchikey: str
    morgan_fingerprint: str
    mn_node_id: str
    adduct: str
    mass_cluster: None | int = None


def find_db_matches(
    mn: nx.Graph, nodes: set, config: SNAPMSConfig, atlas_con: duckdb.DuckDBPyConnection
) -> list[DBMatch]:
    matches = compute_adduct_matches(mn, nodes, config, atlas_con)

    # nodes with the same or very similar masses lead to multiple copies of compounds, here we merge them into one
    matches = merge_duplicates(matches)

    matches = assign_mass_cluster(matches)
    return matches


def build_molecular_families(
    matches: list[DBMatch], config: SNAPMSConfig
) -> nx.Graph | None:
    edges = get_edges(matches, cutoff=config.cutoff)

    # makes sure nodes aren't connected to themselves
    edges = remove_self_similar_vals(edges)

    # snapms logic dictates compounds from the same origin node cannot connect to each other
    edges = remove_same_mass_edges(edges, matches)

    G = nx.Graph()
    G.add_nodes_from((i, asdict(match)) for i, match in enumerate(matches))
    G.add_edges_from(edges)

    remove_small_subgraphs(G, config)  # small families are likely irrelevant

    if len(G) == 0:  # empty graphs are not saved for ease of user investigation
        return None

    add_cluster_numbering(G)
    add_top_candidate_annotation(G)

    return G


# ATLAS
def import_atlas(config: SNAPMSConfig) -> duckdb.DuckDBPyConnection:
    db_path = Path(config.reference_db)

    if not db_path.exists():
        raise FileNotFoundError(f"Reference DB file not found at: {db_path.resolve()}")

    con = duckdb.connect()
    con.execute(
        f"CREATE TABLE df AS SELECT * FROM read_json_auto('{db_path.as_posix()}')"
    )
    con.execute("CREATE INDEX idx_val ON df(neutral_mass)")

    return con


ADDUCT_DICT = {
    "m_plus_h": "[M+H]+",
    "m_plus_na": "[M+Na]+",
    "m_plus_nh4": "[M+NH4]+",
    "m_plus_h_minus_h2o": "[M-H2O+H]+",
    "m_plus_k": "[M+K]+",
    "2m_plus_h": "[2M+H]+",
    "2m_plus_na": "[2M+Na]+",
}


@dataclass
class Adduct:
    multiplier: int  # Oligomer multiplier (e.g., 1 for [M+H]+, 2 for [2M+H]+)
    mass_shift: float  # Net mass added or lost (in Da)
    charge: int  # Absolute charge |z|


# Monoisotopic constants and electron mass
ELECTRON_MASS = 0.00054858
H = 1.007825032 - ELECTRON_MASS
Na = 22.98976928 - ELECTRON_MASS
K = 38.96370668 - ELECTRON_MASS
NH4 = 18.03437413 - ELECTRON_MASS
Ca = 39.9625909 - (2 * ELECTRON_MASS)  # 2+ charge
H2O = 18.010565

# Pre-calculated net mass offsets (mass_shift = adduct_mass - lost_neutral_mass)
ADDUCT_RULES: dict[str, Adduct] = {
    # Single charge [M+...]
    "[M]+": Adduct(multiplier=1, mass_shift=0.0, charge=1),
    "[M+H]+": Adduct(multiplier=1, mass_shift=H, charge=1),
    "[M+Na]+": Adduct(multiplier=1, mass_shift=Na, charge=1),
    "[M+K]+": Adduct(multiplier=1, mass_shift=K, charge=1),
    "[M+NH4]+": Adduct(multiplier=1, mass_shift=NH4, charge=1),
    "[M-H2O]+": Adduct(multiplier=1, mass_shift=-H2O, charge=1),
    "[M-H2O+H]+": Adduct(multiplier=1, mass_shift=H - H2O, charge=1),
    "[M+H-H2O]+": Adduct(multiplier=1, mass_shift=H - H2O, charge=1),
    "[M-2H2O+H]+": Adduct(multiplier=1, mass_shift=H - 2 * H2O, charge=1),
    "[M+H-2H2O]+": Adduct(multiplier=1, mass_shift=H - 2 * H2O, charge=1),
    "[M-3H2O+H]+": Adduct(multiplier=1, mass_shift=H - 3 * H2O, charge=1),
    "[M+CH3CN+H]+": Adduct(multiplier=1, mass_shift=41.026549 + H, charge=1),
    "[M+CH3OH+H]+": Adduct(multiplier=1, mass_shift=32.026215 + H, charge=1),
    "[M+IsoProp+H]+": Adduct(multiplier=1, mass_shift=60.057515 + H, charge=1),
    "[M+2Na-H]+": Adduct(
        multiplier=1, mass_shift=2 * Na - (1.007825032 + ELECTRON_MASS), charge=1
    ),
    "[M+2K-H]+": Adduct(
        multiplier=1, mass_shift=2 * K - (1.007825032 + ELECTRON_MASS), charge=1
    ),
    # Doubly charged [M...]2+
    "[M+2H]2+": Adduct(multiplier=1, mass_shift=2 * H, charge=2),
    "[M+H+H]2+": Adduct(multiplier=1, mass_shift=2 * H, charge=2),
    "[M+2Na]2+": Adduct(multiplier=1, mass_shift=2 * Na, charge=2),
    "[M+Ca]2+": Adduct(multiplier=1, mass_shift=Ca, charge=2),
    "[M+H+NH4]2+": Adduct(multiplier=1, mass_shift=H + NH4, charge=2),
    # Dimers [2M+...]
    "[2M+H]+": Adduct(multiplier=2, mass_shift=H, charge=1),
    "[2M+Na]+": Adduct(multiplier=2, mass_shift=Na, charge=1),
    "[2M+K]+": Adduct(multiplier=2, mass_shift=K, charge=1),
    "[2M-H2O+H]+": Adduct(multiplier=2, mass_shift=H - H2O, charge=1),
    "[2M-2H2O+H]+": Adduct(multiplier=2, mass_shift=H - 2 * H2O, charge=1),
    "[2M+NH3+H]+": Adduct(multiplier=2, mass_shift=17.026549 + H, charge=1),
    "[2M+Ca]2+": Adduct(multiplier=2, mass_shift=Ca, charge=2),
    # Trimers [3M+...]
    "[3M+Ca]2+": Adduct(multiplier=3, mass_shift=Ca, charge=2),
}


def derive_neutral_mass(precursor_mz: float, adduct: str) -> float:
    """Calculate neutral mass from precursor m/z and adduct formula."""
    resolved_adduct = ADDUCT_DICT.get(adduct, adduct)

    rule = ADDUCT_RULES.get(resolved_adduct)
    if not rule:
        raise ValueError(f"Adduct '{adduct}' not recognized")

    return ((precursor_mz * rule.charge) - rule.mass_shift) / rule.multiplier


ADDUCT_ALIASES = ["adduct", "ion"]


def get_adducts(mn: nx.Graph, node: str | int, config: SNAPMSConfig) -> list[str]:
    if not config.detect_adduct:
        return config.adduct_list

    for adduct_key in ADDUCT_ALIASES:
        if adduct_key not in mn.nodes[node]:
            continue

        return list(set([mn.nodes[node][adduct_key]] + config.adduct_list))

    print(
        f"WARNING: no adduct found, relying on default adducts... {mn.nodes[node] = }"
    )
    return config.adduct_list


# DATABASE MATCHING
def compute_adduct_matches(
    mn: nx.Graph,
    nodes: set,
    config: SNAPMSConfig,
    db_con: duckdb.DuckDBPyConnection,
) -> list[DBMatch]:
    result: list[DBMatch] = []

    for node in nodes:
        try:
            precursor_mass = float(mn.nodes[node]["precursor_mz"])
        except KeyError:
            print(
                f"WARNING: precursor mass not found {mn.nodes[node] = }, ignoring this mass..."
            )
            continue

        for adduct in get_adducts(mn, node, config):
            try:
                neutral_mass = derive_neutral_mass(precursor_mass, adduct)
            except ValueError:
                print(f"WARNING: unknown adduct {adduct}, ignoring this mass...")
                continue

            mass_error = round((neutral_mass * config.ppm_error) / 1e6, 4)
            db_matches = search_db(db_con, neutral_mass, mass_error)

            if db_matches.empty:
                continue

            records: list[dict[str, Any]] = db_matches.to_dict("records")
            for record in records:
                match = DBMatch(mn_node_id=node, adduct=adduct, **record)
                result.append(match)

    return result


def search_db(
    db_con: duckdb.DuckDBPyConnection, neutral_mass: float, mass_error: float
):
    lower_bound = neutral_mass - mass_error
    upper_bound = neutral_mass + mass_error

    db_matches = db_con.execute(
        """
        SELECT neutral_mass, smiles, inchikey, morgan_fingerprint
        FROM df
        WHERE neutral_mass BETWEEN ? AND ?
        """,
        [lower_bound, upper_bound],
    ).df()
    return db_matches


def merge_duplicates(matches: list[DBMatch]) -> list[DBMatch]:

    duplicates = defaultdict(list)
    parent_nodes = set()

    for match in matches:
        duplicates[match.smiles].append(match)
        parent_nodes.add(match.mn_node_id)

    result = []
    for smiles, duplicate_matches in duplicates.items():
        m = duplicate_matches[0]

        smiles_parent_nodes = [d.mn_node_id for d in duplicate_matches]
        m.mn_node_id = ";".join({str(n) for n in smiles_parent_nodes})

        result.append(m)

    return result


def group_by_property(
    graph: nx.Graph, key_property: str | int
) -> dict[str | float | int, dict]:

    data = dict(graph.nodes(data=True))
    groups = {}

    for name, metadata in data.items():
        key = metadata[key_property]
        if key in groups:
            groups[key] |= {name: metadata}
        else:
            groups[key] = {name: metadata}

    return groups


def filter_clusters(
    clusters: dict, config: SNAPMSConfig
) -> dict[str | float | int, dict]:
    clusters = {k: v for k, v in clusters.items() if len(v) >= config.min_cluster_size}
    clusters = {k: v for k, v in clusters.items() if len(v) <= config.max_cluster_size}
    return clusters


ID_COUNTER = -1


def get_unique_id() -> int:
    global ID_COUNTER
    ID_COUNTER += 1
    return ID_COUNTER


def get_edges(matches: list[DBMatch], cutoff: float = 0.66) -> list[tuple[int, int]]:
    fingerprints = []
    for match in matches:
        fp = ExplicitBitVect(2048)
        fp.FromBase64(match.morgan_fingerprint)
        fingerprints.append(fp)

    dice_matrix = np.zeros(shape=(len(matches), len(matches)))
    for i, fp in enumerate(fingerprints):
        dice_matrix[i, i + 1 :] = DataStructs.BulkDiceSimilarity(
            fp, fingerprints[i + 1 :]
        )

    rows, cols = np.where(np.triu(dice_matrix, k=1) > cutoff)

    result = []
    for u, v in zip(rows, cols):
        if matches[u].neutral_mass == matches[v].neutral_mass:
            continue

        result.append((u, v))

    return result


def remove_self_similar_vals(edges: list[tuple]) -> list[tuple]:
    return [(u, v) for u, v in edges if u != v]


def remove_same_mass_edges(edges: list[tuple], matches: list[DBMatch]) -> list[tuple]:
    return [
        (u, v) for u, v in edges if matches[u].mass_cluster != matches[v].mass_cluster
    ]


def remove_small_subgraphs(graph: nx.Graph, config: SNAPMSConfig) -> None:
    clusters = [x for x in nx.connected_components(graph)]
    for nodes in clusters:
        if len(nodes) < config.min_annotation_size:
            graph.remove_nodes_from(nodes)


def add_top_candidate_annotation(graph: nx.Graph) -> None:
    clusters = [x for x in nx.connected_components(graph)]
    counts = [_nr_of_unique_compounds(graph, c) for c in clusters]

    for c, nodes in zip(counts, clusters):
        for node in nodes:
            if max(counts) <= 2:
                graph.nodes[node]["is_top_candidate"] = False
            else:
                graph.nodes[node]["is_top_candidate"] = c == max(counts)

            graph.nodes[node]["ann_mass_diversity"] = c


def _nr_of_unique_compounds(graph: nx.Graph, nodes: set) -> int:
    all_mass_clusters = set()
    for node in nodes:
        mass_cluster = graph.nodes[node]["mass_cluster"]
        all_mass_clusters.add(mass_cluster)
    return len(all_mass_clusters)


def assign_mass_cluster(matches: list[DBMatch]) -> list[DBMatch]:
    # This covers an edge case where we may have overlapping ranges of masses.
    # MolA may be queried by node1 & node2, MolB might be queried by node2 and node3 (molB might be just outside the mass range of node1 ).
    # This code merges mass1, mass2 & node3 into ONE group,
    # MolA and MolB are thus counted as the same mass.
    # This prevents edge cases where very similar masses are counted twice due to slightly different mass ranges,
    # which would inflate the mass diversity count where no meaningful diversity exists.

    # Step 1: Pre-parse node sets for each match
    match_nodes = [{int(x) for x in match.mn_node_id.split(";")} for match in matches]

    # Step 2: Merge overlapping sets (Connected Components)
    clusters: list[set[int]] = []
    for nodes in match_nodes:
        # Find all clusters that overlap with the current nodes
        overlapping = [c for c in clusters if c & nodes]

        if not overlapping:
            clusters.append(set(nodes))
        else:
            # Merge current nodes and all overlapping clusters into a single cluster
            merged = nodes.union(*overlapping)
            for c in overlapping:
                clusters.remove(c)
            clusters.append(merged)

    # Step 3: Assign cluster IDs using enumerate()
    for match, query_nodes in zip(matches, match_nodes):
        for i, group in enumerate(clusters):
            if query_nodes & group:
                match.mass_cluster = i
                break
        else:
            assert False, "Match node was not found in any cluster"

    return matches


def add_cluster_numbering(graph: nx.Graph) -> None:
    ordered_clusters = sorted(
        nx.connected_components(graph), key=lambda x: len(x), reverse=True
    )
    for cluster in ordered_clusters:
        i = get_unique_id()
        for node in cluster:
            graph.nodes[node]["parent_cluster"] = i


def group_nodes(graph: nx.Graph) -> defaultdict:
    # TODO: add check that mn_cluster_id corresponds with the acutual cluster sizes
    
    groups: dict[int, list[int]] = defaultdict(list)
    for node in graph:
        mn_cluster_id = graph.nodes[node]["mn_cluster_id"]
        groups[mn_cluster_id].append(node)
    return groups
