import glob
import json
from pathlib import Path
import pprint
from typing import Sequence, Union, Tuple, Dict, Any

import numpy as np
from numpy.random import SeedSequence
from joblib import delayed, Parallel
from scipy import linalg
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn import metrics

from reservoirpy import ESN
from reservoirpy.mat_gen import fast_spectral_initialization, generate_input_weights
from canapy import Config
from canapy.sequence import lcs, lev_sim, group

CONF = Config({
    "N": 1000,
    "sr": 0.7,
    "leak": 9e-2,
    "iss": 1e-3,
    "isd": 5e-3,
    "isd2": 5e-3,
    "ridge": 1e-3, #1e-8
    "n_mfcc": 13,
    "mfcc": True,
    "d": True,
    "dd": True,
    "seed": 12345689
})


def esn_model(config=CONF,
              seed: int = None
              ) -> ESN:

    sr = config.sr
    iss = config.iss
    isd = config.isd
    isd2 = config.isd2
    N = config.N
    leak = config.leak
    ridge = config.ridge
    n_mfcc = config.n_mfcc
    mfcc = config.mfcc
    d = config.d
    dd = config.dd

    rng = np.random.default_rng(seed)

    Wins = []
    input_bias = True
    if mfcc:
        win = generate_input_weights(N, n_mfcc, iss, proba=0.2,
                                     input_bias=input_bias, seed=rng)
        input_bias = False
        Wins.append(win)
    if d:
        wd = generate_input_weights(N, n_mfcc, isd, proba=0.2,
                                    input_bias=input_bias, seed=rng)
        input_bias = False
        Wins.append(wd)
    if dd:
        wdd = generate_input_weights(N, n_mfcc, isd2, proba=0.2,
                                     input_bias=input_bias, seed=rng)
        input_bias = False
        Wins.append(wdd)

    Win = np.hstack(Wins)

    W = fast_spectral_initialization(N, sr=sr, seed=rng, proba=0.2)

    return ESN(W=W, Win=Win, lr=leak, ridge=ridge)


def compute_states(esn: ESN,
                   x: Sequence[np.ndarray],
                   init_state: np.ndarray = None,
                   to_disk: str = None
                   ) -> Sequence[np.ndarray]:

    all_states = esn.compute_all_states(x, init_state=init_state,
                                        backend="loky", verbose=False)

    if to_disk is not None:
        for i, s in enumerate(all_states):
            np.save(Path(to_disk, str(i)+"-states"), s)

    return all_states


def fit(esn: ESN,
        states: Union[str, Sequence[np.ndarray]],
        targets: Sequence[np.ndarray],
        ridge: float = 0.0,
        bias: bool = True,
        ) -> np.ndarray:

    if isinstance(states, str):
        states = as_memmap(states)

    N = states[0].shape[0] + 1 if bias else states[0].shape[0]
    ridge_id = ridge * np.eye(N, dtype=np.float64)
    YXT, XXT = accumulate_tikhonov_terms(states,
                                         targets,
                                         bias)

    Wout = np.dot(YXT, linalg.inv(XXT + ridge_id))
    esn.Wout = Wout
    return Wout


def accumulate_tikhonov_terms(states: Union[Sequence[np.memmap],
                                            Sequence[np.ndarray]],
                              targets: Sequence[np.ndarray],
                              bias: bool = True
                              ) -> Tuple[np.ndarray, np.ndarray]:

    N = states[0].shape[0] + 1 if bias else states[0].shape[0]
    out_dim = targets[0].shape[1]
    XXT = np.zeros((N, N), dtype=np.float64)
    YXT = np.zeros((out_dim, N), dtype=np.float64)
    for s, y in zip(states, targets):
        if bias is not None:
            X = np.vstack([np.ones((1, s.shape[1])), s])
        XXT += np.dot(X, X.T)
        YXT += np.dot(y.T, X.T)

    return YXT, XXT


def cross_validation_fit(esn: ESN,
                         states: Union[str, Sequence[np.ndarray]],
                         targets: Sequence[np.ndarray],
                         oh_encoder,
                         vocab,
                         ridge: float = 0.0,
                         bias: bool = True,
                         folds: int = 10,
                         to_disk: str = None,
                         ) -> Dict[int, Any]:

    if isinstance(states, str):
        states = as_memmap(states)

    N = states[0].shape[0] + 1 if bias else states[0].shape[0]
    ridge_id = ridge * np.eye(N, dtype=np.float64)

    YXT, XXT = accumulate_tikhonov_terms(states,
                                         targets,
                                         bias)

    report = {}
    kf = KFold(n_splits=folds)
    fold = 1
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

        if to_disk is not None:
            fold_path = Path(to_disk, str(fold))
            if not fold_path.exists():
                fold_path.mkdir(parents=True)

            for i, out in enumerate(outputs):
                save_path = fold_path / f"{i}-out"
                np.save(save_path, out)

        global_rep, group_rep = evaluate(outputs, t_i, oh_encoder, vocab)

        report[fold] = dict(global_report=global_rep,
                            group_report=group_rep)

        fold += 1

        if to_disk:
            with Path(to_disk, "report.json").open("w+") as fp:
                json.dump(report, fp)

    pprint.pprint(report)

    return report


def evaluate(predictions, targets, oh_encoder, vocab):

    all_preds = []; all_targets = []
    group_report = {}
    for i in range(len(predictions)):
        preds = np.argmax(predictions[i], axis=1)
        preds = np.array([vocab[j] for j in preds])
        targs = oh_encoder.inverse_transform(targets[i])

        all_preds.append(preds)
        all_targets.append(targs)

        pred_seq = group(preds, min_frame_nb=2)
        true_seq = group(targs, min_frame_nb=2)

        levs = lev_sim(pred_seq, true_seq)
        long = lcs(pred_seq, true_seq)

        group_report[i] = dict(
            levsim=levs,
            lcs=long
        )

    all_preds = np.hstack(all_preds)
    all_targets = np.vstack(all_targets)

    f1 = metrics.f1_score(all_targets,
                          all_preds,
                          average="macro")
    accuracy = metrics.accuracy_score(all_targets,
                                      all_preds)

    global_report = dict(
        f1=f1,
        accuracy=accuracy
    )

    return global_report, group_report


def predict(esn: ESN,
            states: Union[str, Sequence[np.ndarray]],
            ) -> Sequence[np.ndarray]:

    if isinstance(states, str):
        states = as_memmap(states)

    with Parallel(n_jobs=8, backend="loky") as parallel:
        outputs = parallel(
            (delayed(esn.compute_outputs)([s], verbose=False)
             for s in states
             )
        )
    return [out[0].T for out in outputs]


def as_memmap(directory: str) -> Sequence[np.memmap]:
    arrays_files = sorted(glob.glob(str(directory) + "/*.npy"), key=sort_key)
    arrays = []
    for f in arrays_files:
        arrays.append(np.load(f, memmap_mode="r+"))
    return arrays


def load_arrays(directory: str) -> Sequence[np.ndarray]:
    arrays_files = sorted(glob.glob(str(directory) + "/*.npy"), key=sort_key)
    arrays = []
    for f in arrays_files:
        arrays.append(np.load(f))
    return arrays

def sort_key(file_name):
    return int(file_name.split("/")[-1].split("-")[0])


