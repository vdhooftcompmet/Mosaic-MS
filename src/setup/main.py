import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from setup.downloads import download_from_link, ask_for_download
from setup.databases import sdf_to_structure_db, create_databse_intersection_mgf, extract_positive_mode
from setup.mockdata import clean_mgf, filter_by_cosine, filter_by_inchikey
from setup.paths import (
    COCONUT_DB_PATH, COCONUT_DOWNLOAD_LINK, 
    MSNLIB_DB_PATH, MSNLIB_DOWNLOAD_LINK, 
    MS2DEEPSCORE_MODEL_PATH, MS2DEEPSCORE_DOWNLOAD_LINK, 
    SPEC2VEC_MODEL_PATH, SPEC2VEC_DOWNLOAD_LINK, 
    COCONUT_SDF_PATH, 
    MSNLIB_DB_PATH_POS, 
    SUPPORTING_FOLDER, MOCK_DATA_FOLDER, 
    COCONUT_MSNLIB_INTERSECTION, CLEANED_INTERSECTION, 
    INCHIKEY_FILTERED_INTERSECTION, COSINE_FILTERED_INTERSECTION, 
    SPEC2VEC_NEG_PATH, SPEC2VEC_VECTORS_PATH
)
from pathlib import Path


def main():

    Path(SUPPORTING_FOLDER).mkdir(exist_ok=True)
    Path(MOCK_DATA_FOLDER).mkdir(exist_ok=True)
        

    required_downloads = {
        COCONUT_SDF_PATH        : COCONUT_DOWNLOAD_LINK, 
        MSNLIB_DB_PATH          : MSNLIB_DOWNLOAD_LINK, 
        MS2DEEPSCORE_MODEL_PATH : MS2DEEPSCORE_DOWNLOAD_LINK, 
        SPEC2VEC_MODEL_PATH     : SPEC2VEC_DOWNLOAD_LINK
    }
    missing_downloads = {path: link for path, link in required_downloads.items() if not Path(path).exists()}
    download_orders   = {path: link for path, link in missing_downloads.items()  if ask_for_download(link, path)}

    if COCONUT_SDF_PATH in download_orders:
        files = download_from_link(COCONUT_DOWNLOAD_LINK, SUPPORTING_FOLDER, extract=True)

        assert len(files) == 1
        assert files[0].endswith(".sdf")

        sdf_file = files[0]
        Path(sdf_file).rename(COCONUT_SDF_PATH)


    if MSNLIB_DB_PATH in download_orders:
        files = download_from_link(MSNLIB_DOWNLOAD_LINK, SUPPORTING_FOLDER, extract=False)
        
        assert len(files) == 1
        assert files[0].endswith(".mgf")

        mgf_file = files[0]
        Path(mgf_file).rename(MSNLIB_DB_PATH)

    if MS2DEEPSCORE_MODEL_PATH in download_orders:
        files = download_from_link(MS2DEEPSCORE_DOWNLOAD_LINK, SUPPORTING_FOLDER, extract=False)

        assert len(files) == 1
        assert files[0].endswith(".pt")

        pt_file = files[0]
        Path(pt_file).rename(MS2DEEPSCORE_MODEL_PATH)

    if SPEC2VEC_MODEL_PATH in download_orders:
        files = download_from_link(SPEC2VEC_DOWNLOAD_LINK, SUPPORTING_FOLDER, extract=True)

        folder = files[0]
        neg_file        = Path(folder) / "positive_mode" / "150225_Spec2Vec_pos_CleanedLibraries.model.syn1neg.npy"
        vectors_file    = Path(folder) / "positive_mode" / "150225_Spec2Vec_pos_CleanedLibraries.model.wv.vectors.npy"
        model_file = [f for f in (Path(folder) / "positive_mode").glob('*.model')][0]
        
        Path(model_file).rename(SPEC2VEC_MODEL_PATH)
        Path(neg_file).rename(SPEC2VEC_NEG_PATH)
        Path(vectors_file).rename(SPEC2VEC_VECTORS_PATH)

    if not Path(MSNLIB_DB_PATH_POS).exists():

        if not Path(MSNLIB_DB_PATH).exists():
            print(f"> WARNING: cannot create spectral database '{MSNLIB_DB_PATH_POS}' from source file '{MSNLIB_DB_PATH}' because source file does not exist")
        else:
            print(f"building spectral database '{MSNLIB_DB_PATH_POS}' from '{MSNLIB_DB_PATH}'...")
            extract_positive_mode(MSNLIB_DB_PATH, MSNLIB_DB_PATH_POS)
    

    if not Path(COCONUT_DB_PATH).exists():

        if not Path(COCONUT_SDF_PATH).exists():
            print(f"> WARNING: cannot create structure database '{COCONUT_DB_PATH}' from source file '{COCONUT_SDF_PATH}' because source file does not exist")
        else:
            print(f"building structure database '{COCONUT_DB_PATH}' from '{COCONUT_SDF_PATH}'...")
            sdf_to_structure_db(COCONUT_SDF_PATH, COCONUT_DB_PATH)
            

    if not Path(COCONUT_MSNLIB_INTERSECTION).exists():

        missing_source_files = [f for f in [MSNLIB_DB_PATH_POS, COCONUT_DB_PATH] if not Path(f).exists()]
        if missing_source_files:
            print(f"> WARNING: cannot create intersection file '{COCONUT_MSNLIB_INTERSECTION}' from source files because  following source files do not exist: ")
            for f in missing_source_files:
                print(f"'{f}'")
        else:
            print(f"creating intersection file '{COCONUT_MSNLIB_INTERSECTION}' from files '{MSNLIB_DB_PATH_POS}' and '{COCONUT_DB_PATH}'...")
            create_databse_intersection_mgf(MSNLIB_DB_PATH_POS, COCONUT_DB_PATH, COCONUT_MSNLIB_INTERSECTION)


    if not Path(CLEANED_INTERSECTION).exists():

        if not Path(COCONUT_MSNLIB_INTERSECTION).exists():
            print(f"> WARNING: cannot cleaned file '{CLEANED_INTERSECTION}' from source file '{COCONUT_MSNLIB_INTERSECTION}' because source file does not exist")
        else:
            clean_mgf(COCONUT_MSNLIB_INTERSECTION, CLEANED_INTERSECTION)


    if not Path(INCHIKEY_FILTERED_INTERSECTION).exists():

        if not Path(CLEANED_INTERSECTION).exists():
            print(f"> WARNING: cannot cleaned file '{INCHIKEY_FILTERED_INTERSECTION}' from source file '{CLEANED_INTERSECTION}' because source file does not exist")
        else:
            filter_by_inchikey(CLEANED_INTERSECTION, INCHIKEY_FILTERED_INTERSECTION)

