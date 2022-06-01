from pathlib import Path
import json

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from reservoirpy.hyper import research
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Masking
from sklearn.model_selection import KFold

from extract import fetch
from hyperplot import plot_hyperopt_report
from lstm_model import CanaryMFCC, INPUT_DIM, MAX_SEQ_LEN, check_files

BIRD = "marron1"
DATA = Path("data/processed/8kHz-13mfcc", BIRD)
EXP = "lstm-8kHz-13mfcc+delta+delta2"
REPORT = Path("hyper", EXP)

REBOOT = False

N_MFCC = 13

if not REPORT.exists():
    REPORT.mkdir(parents=True)

HP_CONF = {
    "exp": EXP,
    "hp_max_evals": 50,
    "hp_method": "random",
    "seed": 84948631,
    "instances_per_trial": 3,
    "mfcc": True,
    "d": True,
    "dd": True,
    "n_mfcc": N_MFCC,
    "hp_space": {
        "ridge": ["loguniform", 1e-8, 1.],
        "lr": ["loguniform", 1e-5, 1e-1]
    }
}

"""
def objective(dataset, config, *, lr, ridge):

    files = dataset

    losses = []; accuracies = []
    for i in range(config["instances_per_trial"]):

        folds = KFold(n_splits=3)

        for i, (train_idx, val_idx) in enumerate(folds.split(files)):

            lstm = tf.keras.models.Sequential([
                Input((MAX_SEQ_LEN, INPUT_DIM)),
                Masking(mask_value=0.0),
                LSTM(72, return_sequences=True),
                TimeDistributed(
                    Dense(len(targets_names),
                          activation="softmax",
                          kernel_regularizer=tf.keras.regularizers.l2(l2=ridge))
                )
            ])

            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            lstm.compile(optimizer=optimizer,
                         loss="categorical_crossentropy",
                         metrics=["accuracy"])

            train_files = [files[j] for j in train_idx]
            val_files = [files[j] for j in val_idx]
            train_files, val_files = check_files(train_files), check_files(val_files)

            train_generator = CanaryMFCC(train_files, batch_size=32)
            val_generator = CanaryMFCC(val_files, batch_size=32)

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5, verbose=0, restore_best_weights=True)

            history = lstm.fit(
                train_generator, validation_data=val_generator, epochs=50,
                callbacks=[early_stop], verbose=0)

            scores = lstm.evaluate(val_generator, batch_size=32, verbose=0)

            losses.append((scores[0]))
            accuracies.append((scores[0]))

    return {"loss": np.mean(losses), "Accuracy": np.mean(accuracies)}
"""


def objective(dataset, config, *, lr, ridge):

    files = dataset

    with Parallel(n_jobs=config["instances_per_trial"]) as parallel:

        results = parallel(
            delayed(train_instance)(files, ridge, lr)
            for i in range(config["instances_per_trial"])
        )

    return {"loss": np.mean([r[0] for r in results]), "Accuracy": np.mean([r[1] for r in results])}


def train_instance(files, ridge, lr):

    folds = KFold(n_splits=3)

    losses = []; accuracies = []
    for i, (train_idx, val_idx) in enumerate(folds.split(files)):
        lstm = tf.keras.models.Sequential([
            Input((MAX_SEQ_LEN, INPUT_DIM)),
            Masking(mask_value=0.0),
            LSTM(72, return_sequences=True),
            TimeDistributed(
                Dense(len(targets_names),
                      activation="softmax",
                      kernel_regularizer=tf.keras.regularizers.l2(l2=ridge))
            )
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        lstm.compile(optimizer=optimizer,
                     loss="categorical_crossentropy",
                     metrics=["accuracy"])

        train_files = [files[j] for j in train_idx]
        val_files = [files[j] for j in val_idx]
        train_files, val_files = check_files(train_files), check_files(val_files)

        train_generator = CanaryMFCC(train_files, batch_size=32)
        val_generator = CanaryMFCC(val_files, batch_size=32)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, verbose=0, restore_best_weights=True)

        history = lstm.fit(
            train_generator, validation_data=val_generator, epochs=50,
            callbacks=[early_stop], verbose=0)

        scores = lstm.evaluate(val_generator, batch_size=32, verbose=0)

        losses.append((scores[0]))
        accuracies.append((scores[0]))

    return losses, accuracies


if __name__ == "__main__":

    if REBOOT or not (REPORT / "results").exists():
        dataset = fetch(BIRD)
        targets_names = dataset.vocab
        train_songs, _, _ = dataset.to_trainset()

        data = list(train_songs.groupby("wave").groups.keys())

        with (REPORT / "hpconfig.json").open("w+") as fp:
            json.dump(HP_CONF, fp)

        best = research(objective, data, str(REPORT / "hpconfig.json"), "hyper")

    fig = plot_hyperopt_report(f"hyper/{EXP}", ["lr", "ridge"], metric="Accuracy")
    plt.show()
