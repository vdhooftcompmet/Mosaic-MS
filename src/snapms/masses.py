import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import pandas as pd
import networkx as nx

from collections import defaultdict

from argparse import Namespace
from utils.constants import *


# ATLAS
def import_atlas(params: Namespace):
    input_df = pd.read_json(str(params.reference_db), lines=True)

    return input_df


ADDUCT_DICT = {
     "m_plus_h"           : "[M+H]+"     ,  
     "m_plus_na"          : "[M+Na]+"    , 
     "m_plus_nh4"         : "[M+NH4]+"   , 
     "m_plus_h_minus_h2o" : "[M-H2O+H]+" , 
     "m_plus_k"           : "[M+K]+"     , 
     "2m_plus_h"          : "[2M+H]+"    , 
     "2m_plus_na"         : "[2M+Na]+"   , 
}


def derive_neutral_mass(precursor_mz, adduct):
    if adduct in ADDUCT_DICT:
        adduct = ADDUCT_DICT[adduct]

    C  = 12.011
    H  = 1.0080
    O  = 15.999
    N  = 14.007
    Na = 22.989218
    Ca = 40.078
    K  = 38.963158
    IsoProp = 60.09
    
    match adduct:
        case "[M+H]+":
            return precursor_mz - H
        case "[M+Na]+":
            return precursor_mz - Na
        case "[M+NH4]+":
            return precursor_mz - N - H*4
        case "[M-H2O+H]+":
            return precursor_mz + (H*2 + O) - H
        case "[M+K]+":
            return precursor_mz - K
        case "[2M+H]+":
            return (precursor_mz - H) / 2
        case "[2M+Na]+":
            return (precursor_mz - Na) / 2
        case "[M+H+H]2+":
            return (precursor_mz * 2) - H*2
        case "[M-2H2O+H]+":
            return precursor_mz + (H*2 + O*2) - H
        case "[M]+":
            return precursor_mz
        case "[M-H2O]+":
            return precursor_mz + (H*2 + O)
        case "[M+H-H2O]+":
            return precursor_mz + (H*2 + O) - H
        case "[M+H-2H2O]+":
            return precursor_mz + (H*2 + O*2) - H
        case "[M+NH3+H]+":
            return precursor_mz - (N + H*3) - H
        case "[M+2H]2+":
            return (precursor_mz * 2) - (H*2)
        case "[M+CH3CN+H]+":
            return precursor_mz - (C + H*3 + C + N)
        case "[2M-2H2O+H]+":
            return (precursor_mz - (H*2 + O*2) - H ) / 2
        case "[2M+Ca]2+":
            return ((precursor_mz * 2) - Ca ) / 2
        case "[M+CH3OH+H]+":
            return precursor_mz - (C + H*3 + O + H) + H
        case "[M+2Na]2+":
            return (precursor_mz * 2) - (Na*2)
        case "[M+Ca]2+":
            return (precursor_mz * 2) - Ca
        case "[M+CH3OH+H]+":
            return precursor_mz - (C + H*3 + O + H) - H
        case "[M+2Na-H]+":
            return precursor_mz - (Na*2) - H
        case "[2M+NH3+H]+":
            return (precursor_mz - (N + H*3) - H ) / 2
        case "[M+IsoProp+H]+":
            return precursor_mz - IsoProp - H
        case "[3M+Ca]2+":
            return ((precursor_mz * 2) - Ca ) / 3
        case "[2M-H2O+H]+":
            return (precursor_mz + (H*2 + O) - H) / 2
        case "[M+2K-H]+":
            return precursor_mz - (K*2) + H
        case "[2M+K]+":
            return (precursor_mz - K) / 2
        case "[M+H+NH4]2+":
            return (precursor_mz * 2) - H - (N + H*4)
        case "[M-3H2O+H]+":
            return precursor_mz - H + 3*(H*2 + O)
        
    raise ValueError("Adduct not recognized")


def get_adducts(mn, node, params):
    if not params.detect_adduct:
        return params.adduct_list
    
    for adduct_key in ADDUCT_ALIASES:

        if adduct_key not in mn.nodes[node]:
            continue

        return list(set( [mn.nodes[node][adduct_key]] + params.adduct_list ))

    print(f"WARNING: no adduct found, relying on default adducts... {mn.nodes[node] = }")
    return params.adduct_list


# DATABASE MATCHING
def compute_adduct_matches(mn, nodes: dict, params: Namespace, db_df: pd.DataFrame) -> list[dict]:
    result = []

    for node in nodes:

        for adduct in get_adducts(mn, node, params):

            try:
                precursor_mass = float(mn.nodes[node][KEY_PRECURSOR_MZ])
            except KeyError:
                print(f"WARNING: precursor mass not found {mn.nodes[node] = }, ignoring this mass...")
                continue

            try:
                neutral_mass = derive_neutral_mass(precursor_mass, adduct)
            except ValueError:
                print(f"WARNING: unknown adduct {adduct}, ignoring this mass...")
                continue

            try:
                motifs = mn.nodes[node]["motifs"]
            except KeyError as e:
                motifs = ""

            mass_error = round((neutral_mass * params.ppm_error) / 1e6, 4)
            
            mask       = db_df[KEY_NEUTRAL_MASS].between(neutral_mass - mass_error, neutral_mass + mass_error)
            db_matches = db_df[mask]

            if db_matches.empty: 
                continue

            db_matches = db_matches[[KEY_NEUTRAL_MASS, KEY_SMILES, KEY_INCHI_KEY]]

            db_matches[KEY_MN_NODE_ID]      = node
            db_matches[KEY_ADDUCT]          = adduct
            db_matches["motifs"]            = motifs

            result += list(db_matches.to_dict(orient="records"))

    return result


def merge_duplicates(matches: list[dict]):
    
    duplicates = defaultdict(list)
    parent_nodes = set()

    for match in matches:
        smiles = match[KEY_SMILES]
        duplicates[smiles].append(match)

        parent_node = match[KEY_MN_NODE_ID]
        parent_nodes.add(parent_node)

    result = []
    for smiles, duplicate_matches in duplicates.items():
        m = duplicate_matches[0]

        smiles_parent_nodes = [d[KEY_MN_NODE_ID] for d in duplicate_matches]
        m[KEY_MN_NODE_ID] = ";".join({str(n) for n in smiles_parent_nodes})

        shared_motifs = set()
        for d in duplicate_matches:
            for motif in {int(x) for x in d["motifs"].split(";") if x != ""}:
                shared_motifs.add(motif)
        m["motifs"] = ";".join({str(n) for n in shared_motifs})

        for parent_node in parent_nodes:
            represents_node = int(parent_node in smiles_parent_nodes)
            m[f"parent_node_{parent_node}"] = represents_node

        result.append(m)

    return result


def group_by_property(graph: nx.Graph, key_property: str | int) -> dict[dict]:
    data = dict(graph.nodes(data=True))
    groups = {}

    for name, metadata in data.items():
        key = metadata[key_property]
        if key in groups:
            groups[key] |= {name: metadata}
        else:
            groups[key] =  {name: metadata}

    return groups


def filter_clusters(clusters, params) -> dict[dict]:
    clusters = {k: v for k, v in clusters.items() if len(v) >= params.min_cluster_size}
    clusters = {k: v for k, v in clusters.items() if len(v) <= params.max_cluster_size}
    return clusters
