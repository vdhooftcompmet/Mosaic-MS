import tomotopy as tp
from pathlib import Path
from matchms import Spectrum
from matchms.exporting import save_as_mgf


def store_mass2motifs(mass2motifs: list[Spectrum], path):
    save_as_mgf(mass2motifs, str(path), file_mode="w")


def store_model(model: tp.LDAModel, path: Path | str) -> None:
    model.save(str(path))


def load_model(path: Path | str) -> tp.LDAModel:
    path = Path(path)
    model = tp.LDAModel.load(str(path))
    return model
