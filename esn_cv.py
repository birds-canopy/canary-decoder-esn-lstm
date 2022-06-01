import time
import gc
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.random import SeedSequence
from sklearn.model_selection import KFold
from sklearn import metrics
from reservoirpy import ESN
from canapy import Config
from esn_model import esn_model
from extract import fetch
from sequence import word_error_rate, group

REBOOT = True

BIRD = "marron1"
PROCESSED_DATA = Path("data/processed/8kHz-13mfcc-2", BIRD)
REPORTS = Path("reports", BIRD, "esn-4")

if not REPORTS.exists():
    REPORTS.mkdir(parents=True)


SEED = 13245698



def load_data_for_esn(files):
    xs, ys = [], []
    for s in files:
        data = np.load(PROCESSED_DATA / s)
        xs.append(data["x"])
        ys.append(data["y"])
    return xs, ys


def check_files(files):
    new_files = [None] * len(files)
    for i in range(len(files)):
        f = files[i]
        new_files[i] = f + ".npz"
    return new_files


def cross_validation():

    dataset = fetch(BIRD)
    targets_names = dataset.vocab
    train_songs, _, test_songs = dataset.to_trainset()

    files = list(train_songs.groupby("wave").groups.keys())

    print(len(files))

    folds = KFold(n_splits=5)
    rng = np.random.default_rng(SEED)
    seeds = rng.integers(456879, 9999999, 30).tolist()

    for fold, (train_idx, val_idx) in enumerate(folds.split(files)):
        train_files = [files[idx] for idx in train_idx]
        val_files = [files[idx] for idx in val_idx]
        train_files, val_files = check_files(train_files), check_files(val_files)

        X, y = load_data_for_esn(train_files)

        report = REPORTS / "cv" / f"{fold}"
        if not report.exists():
            report.mkdir(parents=True)

        durations = []
        for instance in range(30):
            esn = esn_model(seed=seeds[instance])

            start = time.time()

            esn.train(X, y, backend="loky", use_memmap=True, verbose=True)

            end = time.time()

            print("Total time: ", end - start)

            with (report / f'model-{instance}-seed.json').open("w+") as fp:
                json.dump({"seed": seeds[instance]}, fp)

            durations.append(end - start)

            Xval, yval = load_data_for_esn(val_files)

            val_preds, _ = esn.run(Xval, backend="loky", verbose=True)

            dump_dir = report / f"preds-{instance}"
            if not dump_dir.exists():
                dump_dir.mkdir()

            oh_encoder = dataset.oh_encoder
            mets = {}
            for song, pred, truth in zip(val_files, val_preds, yval):
                mets[song] = {}

                targets = np.vstack(oh_encoder.inverse_transform(truth)).flatten()

                top_1 = np.argmax(pred, axis=1)
                top_1 = np.array([targets_names[t] for t in top_1])

                accuracy = metrics.accuracy_score(targets, top_1)
                mets[song]["accuracy"] = accuracy

                np.savez(str(dump_dir / song), pred=pred, top_1=top_1, targets=targets)

                wers = {}
                for min_frame in range(31):
                    pred_seq = group(top_1, min_frame_nb=min_frame)
                    true_seq = group(targets, min_frame_nb=0)

                    wer = word_error_rate(true_seq, pred_seq)
                    wers[min_frame] = wer

                mets[song]["wer"] = wers

                gc.collect()

            with (report / f"metrics-{instance}.json").open("w+") as fp:
                json.dump(mets, fp)

        with (report / "duration.json").open("w+") as fp:
            json.dump({"duration": np.mean(durations)}, fp)


def complete_training():

    dataset = fetch(BIRD)
    targets_names = dataset.vocab
    train_songs, _, test_songs = dataset.to_trainset()

    files = list(train_songs.groupby("wave").groups.keys())
    test_files = list(test_songs.groupby("wave").groups.keys())

    rng = np.random.default_rng(SEED)
    seeds = rng.integers(456879, 9999999, 30).tolist()

    train_files, val_files = check_files(files), check_files(test_files)

    X, y = load_data_for_esn(train_files)

    report = REPORTS / "complete"
    if not report.exists():
        report.mkdir(parents=True)

    durations = []
    for instance in range(30):
        esn = esn_model(seed=seeds[instance])

        start = time.time()

        esn.train(X, y, backend="loky", use_memmap=True, verbose=True)

        end = time.time()

        print("Total time: ", end - start)

        with (report / f'model-{instance}-seed.json').open("w+") as fp:
            json.dump({"seed": seeds[instance]}, fp)

        durations.append(end - start)

        Xval, yval = load_data_for_esn(val_files)

        val_preds, _ = esn.run(Xval, backend="loky", verbose=True)

        dump_dir = report / f"preds-{instance}"
        if not dump_dir.exists():
            dump_dir.mkdir()

        oh_encoder = dataset.oh_encoder
        mets = {}
        for song, pred, truth in zip(val_files, val_preds, yval):
            mets[song] = {}

            targets = np.vstack(oh_encoder.inverse_transform(truth)).flatten()

            top_1 = np.argmax(pred, axis=1)
            top_1 = np.array([targets_names[t] for t in top_1])

            accuracy = metrics.accuracy_score(targets, top_1)
            mets[song]["accuracy"] = accuracy

            np.savez(str(dump_dir / song), pred=pred, top_1=top_1, targets=targets)

            wers = {}
            for min_frame in range(31):
                pred_seq = group(top_1, min_frame_nb=min_frame)
                true_seq = group(targets, min_frame_nb=0)

                wer = word_error_rate(true_seq, pred_seq)
                wers[min_frame] = wer

            mets[song]["wer"] = wers

            gc.collect()

        with (report / f"metrics-{instance}.json").open("w+") as fp:
            json.dump(mets, fp)

    with (report / "duration.json").open("w+") as fp:
        json.dump({"duration": np.mean(durations)}, fp)


def reduced_songs_training():

    dataset = fetch(BIRD)
    targets_names = dataset.vocab
    train_songs, _, test_songs = dataset.to_trainset()

    files = list(train_songs.groupby("wave").groups.keys())
    test_files = list(test_songs.groupby("wave").groups.keys())

    rng = np.random.default_rng(SEED)
    seeds = rng.integers(456879, 9999999, 30).tolist()

    test_files = check_files(test_files)
    Xval, yval = load_data_for_esn(test_files)

    for max_song in [5, 10, 20, 30, 50, 70]:

        durations = []

        report = REPORTS / "reduced" / f"{max_song}"
        if not report.exists():
            report.mkdir(parents=True)

        for instance in range(30):

            fold_files = rng.choice(files, max_song, replace=False)
            fold_files = check_files(fold_files)

            X, y = load_data_for_esn(fold_files)

            esn = esn_model(seed=seeds[instance])

            start = time.time()

            esn.train(X, y, backend="loky", use_memmap=True, verbose=True)

            end = time.time()

            print("Total time: ", end - start)

            with (report / f'model-seed.json').open("w+") as fp:
                json.dump({"seed": seeds[instance]}, fp)

            durations.append(end - start)

            train_preds, _ = esn.run(X, backend="loky", verbose=True)

            dump_dir = report / f"preds-{instance}"
            if not dump_dir.exists():
                dump_dir.mkdir()

            oh_encoder = dataset.oh_encoder
            mets = {}
            for song, pred, truth in zip(test_files, train_preds, y):
                mets[song] = {}

                targets = np.vstack(oh_encoder.inverse_transform(truth)).flatten()

                top_1 = np.argmax(pred, axis=1)
                top_1 = np.array([targets_names[t] for t in top_1])

                accuracy = metrics.accuracy_score(targets, top_1)
                mets[song]["accuracy"] = accuracy

                np.savez(str(dump_dir / song), pred=pred, top_1=top_1, targets=targets)

                wers = {}
                for min_frame in range(31):
                    pred_seq = group(top_1, min_frame_nb=min_frame)
                    true_seq = group(targets, min_frame_nb=0)

                    wer = word_error_rate(true_seq, pred_seq)
                    wers[min_frame] = wer

                mets[song]["wer"] = wers

            with (report / f"metrics-train-{instance}.json").open("w+") as fp:
                json.dump(mets, fp)

            val_preds, _ = esn.run(Xval, backend="loky", verbose=True)

            dump_dir = report / f"preds-{instance}"
            if not dump_dir.exists():
                dump_dir.mkdir()

            oh_encoder = dataset.oh_encoder
            mets = {}
            for song, pred, truth in zip(test_files, val_preds, yval):
                mets[song] = {}

                targets = np.vstack(oh_encoder.inverse_transform(truth)).flatten()

                top_1 = np.argmax(pred, axis=1)
                top_1 = np.array([targets_names[t] for t in top_1])

                accuracy = metrics.accuracy_score(targets, top_1)
                mets[song]["accuracy"] = accuracy

                np.savez(str(dump_dir / song), pred=pred, top_1=top_1, targets=targets)

                wers = {}
                for min_frame in range(31):
                    pred_seq = group(top_1, min_frame_nb=min_frame)
                    true_seq = group(targets, min_frame_nb=0)

                    wer = word_error_rate(true_seq, pred_seq)
                    wers[min_frame] = wer

                mets[song]["wer"] = wers

            with (report / f"metrics-{instance}.json").open("w+") as fp:
                json.dump(mets, fp)

        with (report / "duration.json").open("w+") as fp:
            json.dump({"duration": np.mean(durations)}, fp)


if __name__ == "__main__":

    # cross_validation()
    # complete_training()
    # reduced_songs_training()

    """
    dataset = fetch(BIRD)
    targets_names = dataset.vocab
    train_songs, _, test_songs = dataset.to_trainset()

    files = list(train_songs.groupby("wave").groups.keys())
    test_files = list(test_songs.groupby("wave").groups.keys())

    files = check_files(files)
    test_files = check_files(test_files)

    X, y = load_data_for_esn(files)
    Xtest, ytest = load_data_for_esn(test_files)

    trdurs = [len(yy) for yy in y]
    tedurs = [len(yy) for yy in ytest]
    trdur = np.sum(trdurs)
    tedur = np.sum(tedurs)

    np.save("train-duration", trdur)
    np.save("test-duration", tedur)

    rng = np.random.default_rng(SEED)
    seeds = rng.integers(456879, 9999999, 30).tolist()

    Xval, yval = load_data_for_esn(test_files)

    durations = []
    for max_song in [5, 10, 20, 30, 50, 70]:

        durs = []
        for instance in range(30):
            fold_files = rng.choice(files, max_song, replace=False)
            fold_files = check_files(fold_files)

            X, y = load_data_for_esn(fold_files)

            dur = []
            for yy in y:
                dur.append(len(yy))
            durs.append(np.sum(dur))
        durations.append(durs)

    np.save("durations", durations)
    """

    dataset = fetch(BIRD)
    targets_names = dataset.vocab
    train_songs, _, test_songs = dataset.to_trainset()

    files = list(train_songs.groupby("wave").groups.keys())
    test_files = list(test_songs.groupby("wave").groups.keys())

    rng = np.random.default_rng(SEED)
    seeds = rng.integers(456879, 9999999, 30).tolist()

    train_files, val_files = check_files(files), check_files(test_files)

    X, y = load_data_for_esn(train_files)

    esn = esn_model(seed=SEED)

    start = time.time()
    states = esn.train(X, y, backend="loky", use_memmap=True, verbose=True)
    end = time.time()

    print("Parallel (12 cores) :", end - start)

    states = np.vstack(states)
    y = np.vstack(y)
    print(states.shape)

    np.save("X_canary.npy", states)
    np.save("Y_canary.npy", y)

    """
    start = time.time()
    states = esn.train(X, y, backend="loky", use_memmap=True, workers=1, verbose=True)
    end = time.time()

    print("Sequential :", end - start)
    """
