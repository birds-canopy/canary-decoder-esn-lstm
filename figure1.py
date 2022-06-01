import glob
import itertools
from pathlib import Path

import librosa as lbr
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from canapy import Dataset
from matplotlib import cm

from extract import fetch

SONG = "455_marron1_May_23_2016_49955646"
BIRD = "marron1"
SAMPLE_BOUNDS = (3, 5.6)
DATA = Path("data/processed/8kHz-13mfcc", BIRD)


def load_audio(dataset: Dataset,
               song: str
               ) -> np.ndarray:
    sr = dataset.config.sampling_rate
    return lbr.load(Path(f"data/marron1/data/{song}.wav"), sr=sr)


def figure1mfcc(dataset: Dataset, song: str) -> plt.Figure:
    wave, sr = load_audio(dataset, song)

    wave = wave[round(SAMPLE_BOUNDS[0] * sr): round(SAMPLE_BOUNDS[1] * sr)]

    config = dataset.config
    df = dataset.df
    n_fft = config.as_fftwindow("n_fft")
    hop_length = config.as_fftwindow("hop_length")
    win_length = config.as_fftwindow("win_length")

    spec = lbr.feature.melspectrogram(y=wave, sr=sr,
                                      n_fft=n_fft,
                                      hop_length=hop_length,
                                      win_length=win_length,
                                      fmin=500,
                                      fmax=8000,
                                      )
    spec = lbr.power_to_db(spec)

    mfcc = lbr.feature.mfcc(S=spec,
                            n_mfcc=13,
                            lifter=40)

    d = lbr.feature.delta(mfcc, mode='wrap')
    dd = lbr.feature.delta(mfcc, order=2, mode='wrap')

    df = df[df["wave"] == song + ".wav"]
    df = df[(df["end"] < SAMPLE_BOUNDS[1]) & (df["start"] > SAMPLE_BOUNDS[0])]

    df["uend"] = np.round(df["end"] * sr - sr * 3).astype("int")
    df["ustart"] = np.round(df["start"] * sr - sr * 3).astype("int")

    df["sstart"] = np.round(df["ustart"] / hop_length).astype("int")
    df["send"] = np.round(df["uend"] / hop_length).astype("int")

    palette = ["gray", "white"]
    fig = plt.figure(figsize=(4.8, 2.4))

    ticks = np.linspace(0, df["send"].values[-1], 7)

    #axspec = fig.add_subplot(411)
    axspec = fig.add_subplot(111)
    axspec.imshow(spec, aspect='auto', origin="lower")
    axspec.set_xticks(ticks)
    axspec.set_xticklabels(np.round(ticks * hop_length / sr, 1).astype(str), size=10)
    axspec.set_yticks([])
    #axspec.set_yticklabels([0, 50, 100], size=10)
    axspec.grid(False)
    axspec.set_ylabel("Me spectrogram.", size=10)
    axspec.set_xlabel("Time (s)", size=10)

    """
    axmfcc = fig.add_subplot(412, sharex=axspec)
    axmfcc.imshow(mfcc, aspect='auto', origin='lower',
                  cmap="copper")
    axmfcc.grid(False)
    axmfcc.set_xticks(ticks)
    axmfcc.set_xticklabels(np.round(ticks * hop_length / sr, 1).astype(str), size=8)
    axmfcc.set_yticks([5, 10, 15, 20])
    axmfcc.set_yticklabels([5, 10, 15, 20], size=8)
    axmfcc.set_ylabel("MFCC", size=10)

    axd = fig.add_subplot(413, sharex=axspec)
    axd.imshow(d, aspect='auto', origin='lower',
               cmap="copper")
    axd.grid(False)
    axd.set_xticks(ticks)
    axd.set_xticklabels(np.round(ticks * hop_length / sr, 1).astype(str), size=8)
    axd.set_yticks([5, 10, 15, 20])
    axd.set_yticklabels([5, 10, 15, 20], size=8)
    axd.set_ylabel(r"$\Delta$", size=10)

    axdd = fig.add_subplot(414, sharex=axspec)
    axdd.imshow(dd, aspect='auto', origin='lower')
    axdd.grid(False)
    axdd.set_xticks(ticks)
    axdd.set_xticklabels(np.round(ticks * hop_length / sr, 1).astype(str), size=8)
    axdd.set_yticks([5, 10, 15, 20])
    axdd.set_yticklabels([5, 10, 15, 20], size=8)
    axdd.set_ylabel(r"$\Delta^2$", size=10)
    axdd.set_xlabel("Time (s)", size=10)
    """

    color_gen = itertools.cycle(palette)
    for syll, send, sstart in zip(df['syll'], df['send'], df['sstart']):
        if syll != "SIL":
            c = next(color_gen)
            axspec.hlines(20, sstart, send, linewidth=5, zorder=4, color=c)
            axspec.axvline(sstart, send, linestyle="--", zorder=4, color="w")
            axspec.text(send - (send - sstart) // 2, 30, syll, color=c, size=10)
            """
            axmfcc.hlines(4, sstart, send, linewidth=5, zorder=4, color=c)
            axmfcc.axvline(sstart, send, linestyle="-", zorder=4, color="w")
            axd.hlines(4, sstart, send, linewidth=5, zorder=4, color=c)
            axd.axvline(sstart, send, linestyle="-", zorder=4, color="w")
            axdd.hlines(4, sstart, send, linewidth=5, zorder=4, color=c)
            axdd.axvline(sstart, send, linestyle="-", zorder=4, color="w")
            """

    plt.tight_layout()

    return fig


def figure1occurences(dataset: Dataset, relative=True) -> plt.Figure:
    df = dataset.df
    df['duration'] = df['end'] - df['start']

    s = pd.DataFrame(df[df["syll"] != "SIL"].groupby("syll").count()["duration"])
    s.columns = ['count']
    s.sort_values(by=["count"], ascending=False, inplace=True)

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(xmargin=0.01)

    plt.xticks(rotation=90)
    xrange = [i * 3 for i in range(len(s))]

    ax.tick_params(labelsize=8)

    if relative:
        s["rel"] = s["count"] / s["count"].sum()
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0, xmax=s["rel"].sum()))
        bars = ax.bar(xrange, s["rel"], 2, color="w", edgecolor="black", linewidth=0.3)
    else:
        bars = ax.bar(xrange, s["count"], 2, color="w", edgecolor="black", linewidth=0.3)

    ax.set_xticks(xrange)

    ax.set_xticklabels(s.index, size=10)

    ax.set_xlabel("Syllables", size=11)
    ax.set_ylabel("Occurrence (%)", size=11)

    df.drop(['duration'], axis=1, inplace=True)

    plt.tight_layout()

    return fig


def figure1durations(dataset: Dataset, relative=True) -> plt.Figure:
    df = dataset.df
    df['duration'] = df['end'] - df['start']

    s = pd.DataFrame(df[df["syll"] != "SIL"].groupby("syll")["duration"].sum())
    s.columns = ['count']
    s.sort_values(by=["count"], ascending=False, inplace=True)

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(xmargin=0.01)

    plt.xticks(rotation=90)
    xrange = [i * 3 for i in range(len(s))]

    ax.tick_params(labelsize=8)

    bars = ax.bar(xrange, s["count"], 2, color="w", edgecolor="black", linewidth=0.3)

    ax.set_xticks(xrange)

    ax.set_xticklabels(s.index, size=10)

    ax.set_xlabel("Syllables", size=11)
    ax.set_ylabel("Duration (s)", size=11)

    df.drop(['duration'], axis=1, inplace=True)

    plt.tight_layout()

    return fig


def figure1mfccvalues():

    inputs = []
    for file in glob.glob(str(DATA) + "/*.npz"):
        data = np.load(file)
        inputs.append(data["x"])

    inputs = np.vstack(inputs)

    n_mfcc = inputs.shape[1] // 3

    fig = plt.figure(figsize=(6, 3))

    plt.tick_params(labelsize=8)

    plt.tick_params(axis="x",
                    which="both",
                    bottom=False,
                    labelbottom=False)

    ax = fig.add_subplot(111)
    ax.boxplot(inputs[:, :n_mfcc],
               showfliers=False,
               whiskerprops={
                   "color": "orange"
               },
               boxprops={
                   "color": "orange"
               },
               positions=np.arange(1, n_mfcc+1))
    ax.boxplot(inputs[:, n_mfcc:2*n_mfcc],
               showfliers=False,
               whiskerprops={
                   "color": "indianred"
               },
               boxprops={
                   "color": "indianred"
               },
               positions=np.arange(n_mfcc+1, 2*n_mfcc+1))
    ax.boxplot(inputs[:, 2*n_mfcc:],
               showfliers=False,
               whiskerprops={
                   "color": "cornflowerblue"
               },
               boxprops={
                   "color": "cornflowerblue"
               },
               positions=np.arange(2*n_mfcc+1, 3*n_mfcc+1))

    ax.axhline(0, color="gray", linewidth=0.5)

    ax.axhline(100, linestyle="--", color="cornflowerblue", linewidth=0.5)
    ax.axhline(-100, linestyle="--", color="cornflowerblue", linewidth=0.5)

    ax.axhline(1000, linestyle="--", color="orange", linewidth=0.5)
    ax.axhline(-1000, linestyle="--", color="orange", linewidth=0.5)

    ax.text(n_mfcc//2 + 1, -950, "MFCC", color="orange")
    ax.text(n_mfcc + n_mfcc//2 + 1, -400, r"$\Delta$", color="indianred")
    ax.text(2*n_mfcc + n_mfcc//2 + 1, -400, r"$\Delta^2$", color="cornflowerblue")

    ticks = [-2000, -1500, -1000, -100, 100, 1000]
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)

    ax.margins(y=0.1)

    plt.tight_layout()

    return fig


if __name__ == "__main__":
    dataset = fetch(BIRD)

    fig = figure1mfcc(dataset, SONG)
    plt.show()
    fig.savefig("figures/figure1mfcc.pdf")
    fig.savefig("figures/figure1mfcc.png")

    fig = figure1occurences(dataset)
    plt.show()
    fig.savefig("figures/occurences.pdf")
    fig.savefig("figures/occurences.png")

    fig = figure1durations(dataset)
    plt.show()
    fig.savefig("figures/durations.pdf")
    fig.savefig("figures/durations.png")

    fig = figure1mfccvalues()
    plt.show()
    fig.savefig("figures/figure1mfccvalues.eps")
    fig.savefig("figures/figure1mfccvalues.png")
