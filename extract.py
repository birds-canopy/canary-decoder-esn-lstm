import itertools
import glob
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
import librosa as lbr
from tqdm import tqdm
from dataset import Dataset, Config

BIRD = "marron1"
OUTPUT_DIR = Path("data/processed/8kHz-13mfcc-2", BIRD)


CONF = Config({
    "sampling_rate": 44100,
    "n_fft": 0.04,
    "hop_length": 0.01,
    "win_length": 0.02,
    "n_mfcc": 13,
    "lifter": 40,
    "fmin": 500,
    "fmax": 8000,
    "mfcc": True,
    "delta": True,
    "delta2": True,
    "padding": "wrap"
})


def fetch(bird: str, path="data") -> Dataset:
    data_path = Path(path, bird)
    return Dataset(data_path)


def to_disk(song: str,
            x: np.ndarray,
            y: np.ndarray,
            wave):

    np.savez(OUTPUT_DIR / song, x=x, y=y, wave=wave)


def preprocess_audio(audio, annots, config=CONF, oh_encoder=None):

    m = lbr.feature.mfcc(audio, config.sampling_rate, n_mfcc=config.n_mfcc,
                         hop_length=config.as_fftwindow("hop_length"),
                         n_fft=config.as_fftwindow("n_fft"),
                         win_length=config.as_fftwindow("win_length"),
                         fmin=config.fmin, fmax=config.fmax,
                         lifter=config.lifter)

    features = []
    if config.mfcc:
        features.append(m)
    if config.delta:
        features.append(lbr.feature.delta(m, mode=config.padding))
    if config.delta2:
        features.append(lbr.feature.delta(m, order=2, mode=config.padding))

    features = np.vstack(features)

    if type(annots.syll) is str:
        teachers = np.tile(annots.syll, m.shape[1])
    else:
        teachers = tile_teachers(annots, config=config)

    # trim
    if teachers.shape[0] <= features.shape[1]:
        features = features[:, :teachers.shape[0]]
    else:
        teachers = teachers[:features.shape[1]]

    if oh_encoder is not None:
        teachers = oh_encoder.transform(teachers.reshape(-1, 1))

    return features.T, teachers


def tile_teachers(annots, config=CONF):
    """Tile teachers labels along time axis.
    """

    starts = np.array([config.frames(s) for s in annots.start]).astype("int")
    ends = np.array([config.frames(e) for e in annots.end]).astype("int")

    y = np.zeros(ends[-1], dtype="U10")
    for s, e, syll in zip(starts, ends, annots.syll):
        y[s: e] = syll

    if (y == '').sum() != 0:
        y[y == ''] = "SIL"

    return y


def preprocess(files, config=CONF):

    Xs, ys, waves = {}, {}, {}

    for wave in tqdm(files):
        song = Path(wave).name
        annotations = df[df["wave"] == song]
        audio, _ = lbr.load(wave, sr=config.sampling_rate)

        X, y = preprocess_audio(audio, annotations, config=config, oh_encoder=dataset.oh_encoder)

        Xs[song] = X
        ys[song] = y
        waves[song] = audio

    return Xs, ys, waves


if __name__ == "__main__":

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    dataset = fetch(BIRD)
    df = dataset.df
    files = glob.glob(f"data/{BIRD}/data/*.wav")

    print(CONF)

    x_dict, y_dict, w_dict = preprocess(files)

    feats = list(x_dict.values())
    lengths = []
    for ft in feats:
        lengths.append(ft.shape[0])

    print(f"Max sequence length: {max(lengths)}")
    print(f"Min sequence length: {min(lengths)}")
    print(f"Mean sequence length: {np.mean(lengths)} +- {np.std(lengths)}")
    print(f"Median sequence length: {np.median(lengths)}")
    quant = np.quantile(lengths, [0.25, 0.75])
    print(f"Q25, Q75, IQR: {quant}, {quant[1] - quant[0]}")

    with Parallel(n_jobs=4) as parallel:
        parallel(delayed(to_disk)(s, x, y_dict[s], w_dict[s])
                 for s, x in x_dict.items())
