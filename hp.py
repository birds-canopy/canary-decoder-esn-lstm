import glob
import json
from typing import Sequence
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from dataset import Config
from sequence import lev_sim, group
from numpy.random import SeedSequence
from reservoirpy import ESN
from reservoirpy.hyper import research
from reservoirpy.activationsfunc import softmax
from sklearn.model_selection import KFold
from sklearn import metrics
from scipy import linalg

from extract import fetch
from esn_model import esn_model, compute_states, accumulate_tikhonov_terms, predict
from hyperplot import plot_hyperopt_report

BIRD = "marron1"
DATA = Path("data/processed/8kHz-13mfcc", BIRD)
EXP = "8kHz-13mfcc+delta+delta2-narrow"
REPORT = Path("hyper", EXP)

NORMALIZE = False
N_MFCC = 13

if not REPORT.exists():
    REPORT.mkdir(parents=True)

HP_CONF = {
    "exp": EXP,
    "hp_max_evals": 300,
    "hp_method": "random",
    "seed": 84948631,
    "instances_per_trial": 5,
    "mfcc": True,
    "d": True,
    "dd": True,
    "n_mfcc": N_MFCC,
    "hp_space": {
        "N": ["choice", 300],
        "sr": ["loguniform", 1e-5, 10],
        "leak": ["choice", 0.09],
        "iss": ["choice", 1e-3],
        "isd": ["choice", 5e-3],
        "isd2": ["loguniform", 1e-5, 10],
        "ridge": ["loguniform", 1e-9, 1]
    }
}


def objective(dataset, config, *, N, sr, leak, iss, isd, isd2, ridge):
    esn_config = Config(dict(
        N=N, leak=leak, sr=sr, iss=iss, isd=isd, isd2=isd2, ridge=ridge,
        seed=config["seed"], n_instances=config["instances_per_trial"],
        n_mfcc=config["n_mfcc"], d=config["d"], dd=config["dd"], mfcc=config["mfcc"],
    ))

    inputs, targets, dataset = dataset

    states, instances = run_instances(inputs, esn_config)

    for i in range(len(states)):
        report = cross_validation_fit(instances[i],
                                      states[i],
                                      targets,
                                      oh_encoder=dataset.oh_encoder,
                                      vocab=dataset.vocab,
                                      ridge=ridge,
                                      folds=5)

    return report


def cross_validation_fit(esn,
                         states,
                         targets,
                         oh_encoder,
                         vocab,
                         ridge: float = 0.0,
                         bias: bool = True,
                         folds: int = 10,
                         ):

    N = states[0].shape[0] + 1 if bias else states[0].shape[0]
    ridge_id = ridge * np.eye(N, dtype=np.float64)

    YXT, XXT = accumulate_tikhonov_terms(states,
                                         targets,
                                         bias)

    reports = []
    kf = KFold(n_splits=folds)
    for train_idx, val_idx in kf.split(states):
        s_i = [states[i] for i in val_idx]
        t_i = [targets[i] for i in val_idx]
        YXTi_v, XXTi_v = accumulate_tikhonov_terms(s_i,
                                                   t_i,
                                                   bias)
        YXTi, XXTi = YXT - YXTi_v, XXT - XXTi_v

        Wout = np.dot(YXTi, linalg.inv(XXTi + ridge_id))
        esn.Wout = Wout

        outputs = predict(esn, s_i)

        global_rep = evaluate(outputs, t_i, oh_encoder, vocab)

        reports.append(global_rep)

    reports = {
        "loss": np.mean([rep["loss"] for rep in reports]),
        "Accuracy": np.mean([rep["accuracy"] for rep in reports]),
        "F1": np.mean([rep["f1"] for rep in reports]),
        "Lev. sim.": np.mean([rep["lev_sim"] for rep in reports]),
    }

    return reports


def evaluate(predictions, targets, oh_encoder, vocab):

    all_preds = []; all_targets = []; all_logits = []
    lev_sims = []
    for i in range(len(predictions)):
        logits = softmax(predictions[i])
        preds = np.argmax(predictions[i], axis=1)
        preds = np.array([vocab[j] for j in preds])
        targs = oh_encoder.inverse_transform(targets[i])

        all_logits.append(logits)
        all_preds.append(preds)
        all_targets.append(targs)

        pred_seq = group(preds, min_frame_nb=2)
        true_seq = group(targs, min_frame_nb=2)

        lev_sims.append(lev_sim(pred_seq, true_seq))

    all_preds = np.hstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_logits = np.vstack(all_logits)

    f1 = metrics.f1_score(all_targets,
                          all_preds,
                          average="macro")
    accuracy = metrics.accuracy_score(all_targets,
                                      all_preds)

    loss = metrics.log_loss(all_targets, all_logits, labels=vocab)

    global_report = dict(
        loss=loss,
        f1=f1,
        accuracy=accuracy,
        lev_sim=np.mean(lev_sims)
    )

    return global_report


def run_instances(inputs, config):
    instances = generate_instances(config.n_instances, config)
    states = []
    for i, esn in enumerate(instances):
        s = compute_states(esn, inputs)
        states.append(s)

    return states, instances


def generate_instances(n_instances: int, config: Config) -> Sequence[ESN]:
    instances = []
    ss = SeedSequence(config["seed"])
    seeds = ss.spawn(n_instances)

    for i in range(n_instances):
        config.seed = seeds[i]
        instances.append(esn_model(config))

    return instances


def normalize(inputs):
    maxs = []; mins = []
    for inp in inputs:
        maxs.append(inp.max(axis=0))
        mins.append(inp.min(axis=0))
    maxs = np.vstack(maxs)
    mins = np.vstack(mins)

    return [2. * (x - mins.min(axis=0)) / (maxs.max(axis=0) - mins.min(axis=0)) - 1.
            for x in inputs]


if __name__ == "__main__":

    rng = np.random.default_rng(289453121)

    if not (REPORT / "results").exists():
        dataset = fetch(BIRD)
        train_songs, _, test_songs = dataset.to_trainset()
        songs = list(train_songs.groupby("wave").groups.keys())
        test_songs = list(test_songs.groupby("wave").groups.keys())

        max_song = 5 * 5

        songs = rng.choice(songs, max_song, replace=False)

        inputs = []
        targets = []
        for file in sorted(glob.glob(str(DATA / "*.npz"))):
            for s in songs:
                if s in file:
                    data = np.load(file)
                    inputs.append(data["x"])
                    targets.append(data["y"])

        test_inputs = []
        test_targets = []
        for file in sorted(glob.glob(str(DATA / "*.npz"))):
            for s in songs:
                if s in file:
                    data = np.load(file)
                    inputs.append(data["x"])
                    targets.append(data["y"])

        if NORMALIZE:
            inputs = normalize(inputs)
        data = inputs, targets, dataset

        with (REPORT / "hpconfig.json").open("w+") as fp:
            json.dump(HP_CONF, fp)

        best = research(objective, data, str(REPORT / "hpconfig.json"), "hyper")

    fig = plot_hyperopt_report(f"hyper/{EXP}", ["sr", "isd2", "ridge"], metric="Accuracy")
    plt.show()
