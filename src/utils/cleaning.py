import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


from mn.bootstrap import clean_mgf
from matchms.exporting import save_as_mgf


def main(params):
    cleaned_spectra = list(clean_mgf(params.raw_mgf))
    save_as_mgf(cleaned_spectra, params.cleaned_mgf)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--raw-mgf", type=str, required=True)
    parser.add_argument("--cleaned-mgf", type=str, required=True)
    params = parser.parse_args()
    main(params)
