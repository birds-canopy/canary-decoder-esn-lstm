import math
import time
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn import metrics
from tensorflow import keras as K
from tensorflow.keras.layers import Input, Dense

from extract import fetch
from sequence import group, word_error_rate

#tf.config.set_visible_devices([], 'GPU')

BIRD = "marron1"
PROCESSED_DATA = Path("data/processed/8kHz-13mfcc-2", BIRD)
REPORT = Path("reports/marron1/mlp")

INPUT_DIM = 13*3
MAX_SEQ_LEN = 5000

SEED = 13245698


def load_data_for_mlp(files):
    xs, ys = [], []
    for s in files:
        data = np.load(PROCESSED_DATA / s)
        xs.append(data["x"])
        ys.append(data["y"])
    return xs, ys


def build_model(targets_names, seed):

    tf.random.set_seed(seed)

    mlp = K.models.Sequential([
        Input((INPUT_DIM,)),
        Dense(500, activation="relu"),
        Dense(len(targets_names),
              activation="softmax",
              kernel_regularizer=K.regularizers.l2(1e-4))
    ])

    adam = K.optimizers.Adam(learning_rate=1e-3)
    mlp.compile(optimizer=adam,
                loss="categorical_crossentropy",
                metrics=["accuracy"])

    return mlp


def check_files(files):
    new_files = [None] * len(files)
    for i in range(len(files)):
        f = files[i]
        if ".wav" in f:
            new_f = str(Path(f).name)
        else:
            new_f = f
        new_files[i] = new_f + ".npz"
    return new_files


def cross_validation():

    dataset = fetch(BIRD)
    targets_names = dataset.vocab
    train_songs, _, test_songs = dataset.to_trainset()

    files = list(train_songs.groupby("wave").groups.keys())

    folds = KFold(n_splits=5)

    rng = np.random.default_rng(SEED)
    seeds = rng.integers(456879, 9999999, 5).tolist()
    for fold, (train_idx, val_idx) in enumerate(folds.split(files)):
        train_files = [files[idx] for idx in train_idx]
        val_files = [files[idx] for idx in val_idx]
        train_files, val_files = check_files(train_files), check_files(val_files)

        X, y = load_data_for_mlp(train_files)
        Xval, yval = load_data_for_mlp(val_files)

        train_x, train_y = np.vstack(X), np.vstack(y)
        val_x, val_y = np.vstack(X), np.vstack(y)

        train_generator = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_generator = train_generator.shuffle(len(train_x),
                                                  seed=seeds[fold],
                                                  reshuffle_each_iteration=True).batch(32)
        val_generator = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        val_generator = val_generator.batch(32)

        durations = []

        report = REPORT / "cv" / f"{fold}"
        if not report.exists():
            report.mkdir(parents=True)

        for instance in range(5):
            mlp = build_model(targets_names, seeds[instance])

            mlp.summary()

            tensorboard = K.callbacks.TensorBoard(log_dir=f'logs-mlp/fold-{fold}-{instance}',
                                                  update_freq=10)
            early_stop = K.callbacks.EarlyStopping(monitor="val_accuracy",
                                                   patience=20,
                                                   restore_best_weights=True)

            start = time.time()

            history = mlp.fit(
                train_generator, validation_data=val_generator, epochs=500,
                callbacks=[tensorboard, early_stop], use_multiprocessing=True,
                workers=12)

            end = time.time()

            print("Total time: ", end - start)

            durations.append(end - start)

            mlp.save(str(report / f'model-{instance}'))

            val_preds = []
            for x_val in Xval:
                val_preds.append(mlp.predict(x_val))

            dump_dir = report / f"preds-{instance}"
            if not dump_dir.exists():
                dump_dir.mkdir()

            oh_encoder = dataset.oh_encoder
            oh_encoder = oh_encoder.set_params(handle_unknown="ignore")
            mets = {}
            for song, pred, truth in zip(val_files, val_preds, yval):
                mets[song] = {}

                raw_targets = np.vstack(oh_encoder.inverse_transform(truth)).flatten()
                targets = raw_targets[raw_targets != None].astype(str)

                top_1 = np.argmax(pred, axis=1)
                top_1 = np.array([targets_names[t] for t in top_1])
                top_1 = top_1[raw_targets != None].astype(str)

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


def complete_training():

    dataset = fetch(BIRD)
    targets_names = dataset.vocab
    train_songs, _, test_songs = dataset.to_trainset()

    train_files = list(train_songs.groupby("wave").groups.keys())
    test_files = list(test_songs.groupby("wave").groups.keys())

    rng = np.random.default_rng(SEED)
    seeds = rng.integers(456879, 9999999, 5).tolist()

    train_files, val_files = check_files(train_files), check_files(test_files)

    X, y = load_data_for_mlp(train_files)
    Xval, yval = load_data_for_mlp(val_files)

    train_generator = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)
    val_generator = tf.data.Dataset.from_tensor_slices((Xval, yval)).batch(32)

    durations = []

    report = REPORT / "complete"
    if not report.exists():
        report.mkdir(parents=True)

    for instance in range(5):
        lstm = build_model(targets_names, seeds[instance])

        lstm.summary()

        tensorboard = K.callbacks.TensorBoard(log_dir=f'logs-3/complete-{instance}',
                                              update_freq=10)
        early_stop = K.callbacks.EarlyStopping(monitor="val_accuracy",
                                               patience=20,
                                               restore_best_weights=True)

        start = time.time()

        history = lstm.fit(
            train_generator, validation_data=val_generator, epochs=500,
            callbacks=[tensorboard, early_stop], use_multiprocessing=True,
            workers=12)

        end = time.time()

        print("Total time: ", end - start)

        durations.append(end - start)

        lstm.save(str(report / f'model-{instance}'))

        val_preds = lstm.predict(Xval)

        dump_dir = report / f"preds-{instance}"
        if not dump_dir.exists():
            dump_dir.mkdir()

        oh_encoder = dataset.oh_encoder
        oh_encoder = oh_encoder.set_params(handle_unknown="ignore")
        mets = {}
        for song, pred, truth in zip(val_files, val_preds, yval):
            mets[song] = {}

            raw_targets = np.vstack(oh_encoder.inverse_transform(truth)).flatten()
            targets = raw_targets[raw_targets != None].astype(str)

            top_1 = np.argmax(pred, axis=1)
            top_1 = np.array([targets_names[t] for t in top_1])
            top_1 = top_1[raw_targets != None].astype(str)

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


def reduced_songs_training():

    dataset = fetch(BIRD)
    targets_names = dataset.vocab
    train_songs, _, test_songs = dataset.to_trainset()

    files = list(train_songs.groupby("wave").groups.keys())
    test_files = list(test_songs.groupby("wave").groups.keys())

    rng = np.random.default_rng(SEED)
    seeds = rng.integers(456879, 9999999, 5).tolist()

    for max_song in [5, 10, 20, 30, 50, 70]:

        durations = []

        report = REPORT / "reduced" / f"{max_song}"
        if not report.exists():
            report.mkdir(parents=True)

        train_files = rng.choice(files, max_song*5, replace=False)
        train_files, val_files = check_files(train_files), check_files(test_files)

        folds = KFold(n_splits=5)

        for instance, (train_idx, _) in enumerate(folds.split(train_files)):
            fold_files = [train_files[idx] for idx in train_idx]

            X, y = load_data_for_mlp(fold_files)
            Xval, yval = load_data_for_mlp(val_files)

            train_generator = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)
            val_generator = tf.data.Dataset.from_tensor_slices((Xval, yval)).batch(32)

            lstm = build_model(targets_names, seeds[instance])

            lstm.summary()

            tensorboard = K.callbacks.TensorBoard(log_dir=f'logs-3/reduced-{max_song}-{instance}',
                                                  update_freq=10)
            early_stop = K.callbacks.EarlyStopping(monitor="val_accuracy",
                                                   patience=20,
                                                   restore_best_weights=True)

            start = time.time()

            history = lstm.fit(
                train_generator, validation_data=val_generator, epochs=500,
                callbacks=[tensorboard, early_stop], use_multiprocessing=True,
                workers=12)

            end = time.time()

            print("Total time: ", end - start)

            durations.append(end - start)

            lstm.save(str(report / f'model-{instance}'))

            val_preds = lstm.predict(Xval)

            dump_dir = report / f"preds-{instance}"
            if not dump_dir.exists():
                dump_dir.mkdir()

            oh_encoder = dataset.oh_encoder
            oh_encoder = oh_encoder.set_params(handle_unknown="ignore")
            mets = {}
            for song, pred, truth in zip(val_files, val_preds, yval):
                mets[song] = {}

                raw_targets = np.vstack(oh_encoder.inverse_transform(truth)).flatten()
                targets = raw_targets[raw_targets != None].astype(str)

                top_1 = np.argmax(pred, axis=1)
                top_1 = np.array([targets_names[t] for t in top_1])
                top_1 = top_1[raw_targets != None].astype(str)

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


            fold_preds = lstm.predict(X)

            dump_dir = report / f"preds-train-{instance}"
            if not dump_dir.exists():
                dump_dir.mkdir()

            mets = {}
            for song, pred, truth in zip(fold_files, fold_preds, y):
                mets[song] = {}

                raw_targets = np.vstack(oh_encoder.inverse_transform(truth)).flatten()
                targets = raw_targets[raw_targets != None].astype(str)

                top_1 = np.argmax(pred, axis=1)
                top_1 = np.array([targets_names[t] for t in top_1])
                top_1 = top_1[raw_targets != None].astype(str)

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

            with (report / "duration.json").open("w+") as fp:
                json.dump({"duration": np.mean(durations)}, fp)


if __name__ == "__main__":

    cross_validation()
    # complete_training()
    # reduced_songs_training()
