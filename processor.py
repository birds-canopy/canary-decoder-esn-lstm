import random
import math

from pathlib import Path

import librosa as lbr
import numpy as np

from tqdm import tqdm


class NoAnnotationsError(Exception):
    pass


class Processor(object):

    # FOR DATA AUGMENTATION ONLY
    # should not be modifiyed

    # reduction factor for random normal distribution std
    RND_NOISE_STD_REDUCTION = 50
    # min random amplitude shift
    RND_AMPLITUDE_LOW = 1.
     # max random amplitude shift
    RND_AMPLITUDE_HIGH = 1.5

    def __init__(self, dataset):

        # connect to dataset
        self.config = dataset.config
        self.dataset = dataset
        self.rs = None

    def __call__(self, df, mode="syn", config=None, return_dict=False):
        """
        Processor object compute features from the given dataframe.
        It is connected to a Dataset instance which give it location
        of audio files.
        """
        return self.preprocess(df, mode, config=config,
                               return_dict=return_dict)

    def _preprocess_audio(self, audio, annots, config=None):
        """
        Preprocess an audio signal :
            - augment if necessary
            - extract and pack MFCC, delta and delta2
            - pack teacher signals
        """
        config = config or self.config

        if hasattr(annots, "augmented"):
            X = self._noisify(audio)
        else:
            X = audio

        m = lbr.feature.mfcc(X, config.sampling_rate, n_mfcc=config.n_mfcc,
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

        if hasattr(annots, "syll"):
            if type(annots.syll) is str:
                teachers = np.tile(annots.syll, m.shape[1])
            else:
                teachers = self._tile_teachers(annots, config=config)

            # trim
            if teachers.shape[0] <= features.shape[1]:
                features = features[:, :teachers.shape[0]]
            else:
                teachers = teachers[:features.shape[1]]

            if self.dataset.oh_encoder is not None:
                teachers = self.dataset.oh_encoder.transform(teachers.reshape(-1, 1))

            return features.T, teachers
        else:
            return features.T, None

    def _tile_teachers(self, annots, config=None):
        """Tile teachers labels along time axis.
        """
        config = config or self.config

        starts = np.array([config.frames(s) for s in annots.start]).astype("int")
        ends = np.array([config.frames(e) for e in annots.end]).astype("int")

        y = np.zeros(ends[-1], dtype="U10")
        for s, e, syll in zip(starts, ends, annots.syll):
            y[s: e] = syll

        if (y == '').sum() != 0:
            y[y == ''] = "SIL"

        return y

    def _extract_phrases(self, audio, annots, config=None):
        """Load all phrases from song.
        """
        config = config or self.config
        for entry in annots.itertuples():
            start, end = config.steps(entry.start), config.steps(entry.end)
            yield audio[start: end], entry

    def _noisify(self, phrase):
        """
        Add random perturbation to an audio signal.
        """
        if self.rs is None:
            self.rs = np.random.RandomState(self.config.seed)
        noise = self.rs.normal(0,
                               random.random()
                               / Processor.RND_NOISE_STD_REDUCTION,
                               size=phrase.shape)
        X = self.rs.uniform(Processor.RND_AMPLITUDE_LOW,
                            Processor.RND_AMPLITUDE_HIGH) * phrase + noise
        return X

    def preprocess(self, df, mode, config=None, return_dict=False):
        """
        Preprocess dataset elements given in 'df', for syn or nsyn mode.
        """
        config = config or self.config
        self.rs = np.random.RandomState(config.seed)

        Xs, ys = [], []
        if return_dict and mode == "syn":
            Xs, ys = {}, {}

        for song in tqdm(df.groupby("wave").groups.keys(),
                         f"Processing data for {mode} training"):
            wave = Path(self.dataset.audiodir, song)
            annotations = df[df["wave"] == song]
            audio, _ = lbr.load(wave, sr=config.sampling_rate)

            if mode == "syn":
                X, y = self._preprocess_audio(audio,
                                              annotations,
                                              config=config)

                if return_dict:
                    Xs[song] = X
                    ys[song] = y
                else:
                    Xs.append(X)
                    ys.append(y)

            elif mode == "nsyn":
                if annotations.get("syll") is not None:
                    for phrase, annots in self._extract_phrases(audio,
                                                                annotations,
                                                                config=config):
                        X, y = self._preprocess_audio(phrase,
                                                      annots,
                                                      config=config)
                        Xs.append(X)
                        ys.append(y)
                else:
                    raise NoAnnotationsError("can't perform nsyn feature "
                                             "extraction if no annotations "
                                             "are provided.")
            else:
                raise ValueError(f"mode: '{mode}' - "
                                 "should be 'syn' or 'nsyn'.")

        return Xs, ys
