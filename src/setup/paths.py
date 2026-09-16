import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from pathlib import Path

# CHANGE THESE TO YOUR OWN PATH
SUPPORTING_FOLDER = Path(r"/lustre/BIF/nobackup/ijcke001/supporting_files")
MOCK_DATA_FOLDER  = Path(r"/lustre/BIF/nobackup/ijcke001/mock_data")


# OTHER PATHS
COCONUT_SDF_PATH                = SUPPORTING_FOLDER / "coconut_sdf.sdf"
COCONUT_DB_PATH                 = SUPPORTING_FOLDER / "coconut_db.jsonl"

GNPS_DB_PATH                    = SUPPORTING_FOLDER / "gnps_db.mgf"
MSNLIB_DB_PATH                  = SUPPORTING_FOLDER / "msnlib_db.mgf"
MSNLIB_DB_PATH_POS              = SUPPORTING_FOLDER / "msnlib_db_positive_mode.mgf"

COCONUT_MSNLIB_INTERSECTION     = SUPPORTING_FOLDER / "coconut_msnlib_intersection.mgf"

SPEC2VEC_MODEL_PATH             = SUPPORTING_FOLDER / "Spec2VecModel.model"
SPEC2VEC_NEG_PATH               = SUPPORTING_FOLDER / "Spec2VecModel.model.syn1neg.npy"
SPEC2VEC_VECTORS_PATH           = SUPPORTING_FOLDER / "Spec2VecModel.model.wv.vectors.npy"
MS2DEEPSCORE_MODEL_PATH         = SUPPORTING_FOLDER / "ms2deepscore_model.pt"

COCONUT_DOWNLOAD_LINK           = r"https://coconut.s3.uni-jena.de/prod/downloads/2026-05/coconut_sdf_2d-05-2026.zip"
SPECTRAL_DB_DOWNLOAD_LINK       = r"https://external.gnps2.org/gnpslibrary/ALL_GNPS.mgf"
MSNLIB_DOWNLOAD_LINK            = r"https://zenodo.org/records/16882111/files/merged_and_cleaned_libraries_1.mgf?download=1"

MS2DEEPSCORE_DOWNLOAD_LINK      = r"https://zenodo.org/records/17826815/files/ms2deepscore_model.pt?download=1"
SPEC2VEC_DOWNLOAD_LINK          = r"https://zenodo.org/records/15857387/files/Spec2Vec.zip?download=1"

ASSETS_FOLDER = Path(__file__).parent.parent.parent / "assets"


CLEANED_INTERSECTION            = MOCK_DATA_FOLDER / "intersection_cleaned.mgf"
COSINE_FILTERED_INTERSECTION    = MOCK_DATA_FOLDER / "intersection_cosine_filtered.mgf"
INCHIKEY_FILTERED_INTERSECTION  = MOCK_DATA_FOLDER / "intersection_inchikey_filtered.mgf"

ANNOTATION_STYLE_FILE           = "../data/snapms_example.cx"
MN_STYLE_FILE                   = "../data/mn_example.cx"
