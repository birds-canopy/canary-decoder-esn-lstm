import itertools
import glob
import shutil
from pathlib import Path
from tqdm import tqdm

from dataset import Dataset

BIRD = "marron1"
OUTPUT_DIR = Path("data/zenodo/", BIRD)
AUDIO_DIR = OUTPUT_DIR / "audio"
ANNOT_DIR = OUTPUT_DIR / "annotations"

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)

if not AUDIO_DIR.exists():
    AUDIO_DIR.mkdir(parents=True)

if not ANNOT_DIR.exists():
    ANNOT_DIR.mkdir(parents=True)


def fetch(bird: str) -> Dataset:
    data_path = Path(f"data/{bird}")
    return Dataset(data_path)


if __name__ == "__main__":

    dataset = fetch(BIRD)
    df = dataset.df
    df = df.drop(["repertoire_file"], axis=1)

    songs = list(df.groupby("wave").groups.keys())

    for s in tqdm(songs):
        d = df[df["wave"] == s]
        d.to_csv(ANNOT_DIR / Path(s).with_suffix(".csv"))

        audio = Path("data") / BIRD / "data" / s
        shutil.copy(audio, AUDIO_DIR)
