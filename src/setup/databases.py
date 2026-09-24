import json
from tqdm import tqdm
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf
from matchms.filtering import derive_ionmode
from rdkit import Chem
from rdkit.Chem import Descriptors


def extract_positive_mode(input_file, output_file):
    spectra = load_from_mgf( str(input_file) )

    result = []
    for spectrum in tqdm(spectra):
        spectrum = derive_ionmode(spectrum)
        if spectrum.get("ionmode") == "positive":
            result.append(spectrum)

    print("saving...")
    save_as_mgf(result, str(output_file), file_mode="w")
    print("saved")


def create_databse_intersection_mgf(spectral_database_mgf, strucuture_database_jsonl, output_file, add_missing=True):
    result = []

    print("calculating structure inchikeys...")
    structure_db_inchi = get_structure_db_inchikeys(strucuture_database_jsonl)

    spectral_db = set_spectral_db_inchikeys(spectral_database_mgf)
    result = [s for s in tqdm(spectral_db) if s.get("inchi_key") in structure_db_inchi]

    save_as_mgf(result, str(output_file), file_mode="w")

    if not add_missing:
        return

    save_as_mgf(spectral_db, str(spectral_database_mgf), file_mode="w")


def sdf_to_structure_db(input_sdf, output_jsonl):
    suppl = Chem.SDMolSupplier( str(input_sdf) )
    seen = set()

    with open( str(output_jsonl), "w", encoding="utf-8") as f:
        for mol in tqdm(suppl):

            if not mol:
                continue

            Chem.RemoveStereochemistry(mol)
            inchi_key = Chem.MolToInchiKey(mol)

            if inchi_key in seen:
                continue

            seen.add(inchi_key)
            record = {
                "smiles": Chem.MolToSmiles(mol),
                "neutral_mass": Descriptors.ExactMolWt(mol),
                "inchikey": inchi_key
            }
            f.write(json.dumps(record) + "\n")


def get_structure_db_inchikeys(strucuture_database_jsonl):
    result = set()

    for line in tqdm(read_structural_db(strucuture_database_jsonl)):
        inchi_key = line.get("inchikey")

        if not inchi_key:
            inchi_key = smiles_to_inchikey(line.get("smiles"))

        if not inchi_key:
            continue

        result.add(inchi_key)

    return result


def read_structural_db(strucuture_database_jsonl):
    with open(strucuture_database_jsonl, 'r') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                yield data

            except json.JSONDecodeError:
                print(f"Error parsing line {i}: {line[:50]}...")
                continue


def set_spectral_db_inchikeys(spectral_database_mgf):
    result = []
    spectra = load_from_mgf( str(spectral_database_mgf) )

    for spectrum in tqdm(spectra):
        inchi_key = spectrum.get("inchikey")

        if not inchi_key:
            smiles = spectrum.get("smiles")
            inchi_key = smiles_to_inchikey(smiles)

        if not inchi_key:
            continue

        spectrum.set("inchikey", inchi_key)

        result.append(spectrum)
    return result


def chache_smiles(fn):
    cache = {}
    def inner(smiles):
        if smiles not in cache:
           cache[smiles] = fn(smiles)
        return cache[smiles]

    return inner


@chache_smiles
def smiles_to_inchikey(smiles):
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    Chem.RemoveStereochemistry(mol)
    inchi_key = Chem.MolToInchiKey(mol)
    return inchi_key
