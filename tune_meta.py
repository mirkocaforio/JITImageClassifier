#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyper-parameter selection of the stacking meta-learner by stratified k-fold
cross-validation on the VAL split (the split the meta-learner is trained on).

Test is never read: every configuration of the grid is scored on the held-out folds of
val, and the best one (lowest mean CV log-loss, a proper scoring rule that rewards good
probabilities and not only the arg-max) is printed as a ready-to-run stacking.py command.

For the boosting models the number of rounds is chosen by early stopping on the held-out
fold, then averaged over the folds: stacking.py trains on the whole val split with that
fixed number of rounds.

Outputs (in --out_dir): tune_<meta_learner>.json with every configuration and the best one.

Run in the JIT conda env:
    python tune_meta.py --preds_dir stacking/preds --meta_learner logreg
    python tune_meta.py --preds_dir stacking/preds --meta_learner xgboost --folds 5
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

from meta_learners import (
    META_LEARNERS,
    best_iteration,
    build_meta_learner,
    check_labels,
    display_name,
    fit_meta_learner,
    meta_features,
    predict_proba_canonical,
)
from stacking import align_split, load_preds

# Max rounds and patience of the early stopping inside the folds (boosting only)
MAX_ROUNDS = {"xgboost": 2000, "catboost": 3000}
PATIENCE = 50

# Search space of each meta-learner, as stacking.py argument names. The trees are invariant
# to the log transform, so `features` is searched for logreg only.
GRIDS = {
    "rf": {
        "max_depth": [None, 6, 8, 10, 12, 16],
        "min_samples_leaf": [1, 5, 10, 20, 50],
        "class_weight": ["balanced", "none"],
    },
    "xgboost": {
        "max_depth": [2, 3, 4, 6],
        "learning_rate": [0.05, 0.1],
        "min_child_weight": [1, 5, 20],
        "class_weight": ["balanced", "none"],
    },
    "catboost": {
        "max_depth": [3, 4, 6],
        "learning_rate": [0.05, 0.1],
        "l2_leaf_reg": [3, 10],
        "class_weight": ["balanced", "none"],
    },
    "logreg": {
        "features": ["probs", "logprobs"],
        "C": [0.01, 0.1, 1, 10, 100],
        "class_weight": ["balanced", "none"],
    },
}


def grid_configs(name):
    keys = list(GRIDS[name])
    return [dict(zip(keys, values)) for values in itertools.product(*GRIDS[name].values())]


def cross_validate(name, config, X_val, y_val, folds, n_classes, args):
    """Mean / std CV log-loss and accuracy of one configuration (+ rounds for boosting)."""
    params = {k: v for k, v in config.items() if k != "features"}
    X = meta_features(X_val, config.get("features", "probs"))
    boosting = name in MAX_ROUNDS
    losses, accs, rounds = [], [], []
    for train_idx, held_idx in folds:
        model = build_meta_learner(
            name, n_classes,
            n_estimators=MAX_ROUNDS[name] if boosting else args.n_estimators,
            early_stopping_rounds=PATIENCE if boosting else None,
            random_state=args.random_state, n_jobs=args.n_jobs,
            **params,
        )
        held = (X[held_idx], y_val[held_idx])
        fit_meta_learner(model, X[train_idx], y_val[train_idx],
                         class_weight=config["class_weight"],
                         eval_set=held if boosting else None)
        if boosting:
            rounds.append(best_iteration(model))
        proba = predict_proba_canonical(model, held[0], n_classes)
        losses.append(log_loss(held[1], proba, labels=range(n_classes)))
        accs.append(accuracy_score(held[1], proba.argmax(1)))
    result = {
        "cv_logloss": float(np.mean(losses)),
        "cv_logloss_std": float(np.std(losses)),
        "cv_accuracy": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs)),
    }
    if boosting:
        result["n_estimators"] = int(round(np.mean(rounds)))
    return result


def stacking_command(name, config, n_estimators, args):
    flags = [f"--meta_learner {name}"]
    for key, value in config.items():
        if value is not None:  # max_depth None = the stacking.py default (unlimited for rf)
            flags.append(f"--{key} {value}")
    if n_estimators is not None:
        flags.append(f"--n_estimators {n_estimators}")
    return f"python stacking.py --preds_dir {args.preds_dir} " + " ".join(flags)


def get_args():
    p = argparse.ArgumentParser("Cross-validated hyper-parameter search of the meta-learner")
    p.add_argument("--preds_dir", default="../datahome/stacking_data/preds", type=str)
    p.add_argument("--out_dir", default="../datahome/stacking_data/", type=str)
    p.add_argument("--meta_learner", default="rf", choices=META_LEARNERS)
    p.add_argument("--folds", default=5, type=int)
    p.add_argument("--n_estimators", default=300, type=int,
                   help="Trees of the forest (rf); the boosting rounds are searched.")
    p.add_argument("--random_state", default=42, type=int)
    p.add_argument("--n_jobs", default=-1, type=int)
    return p.parse_args()


def main():
    args = get_args()
    name = args.meta_learner

    jit_val = load_preds(os.path.join(args.preds_dir, "jit_val.npz"))
    mamba_val = load_preds(os.path.join(args.preds_dir, "mamba_val.npz"))
    X_val, y_val, classes = align_split(jit_val, mamba_val, "val")
    n_classes = len(classes)
    check_labels(y_val, n_classes, "val")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    folds = list(skf.split(X_val, y_val))
    configs = grid_configs(name)
    print(f"[tune] {display_name(name)}: {len(configs)} configurations x {args.folds} folds "
          f"on val {X_val.shape}")

    rows = []
    start = time.time()
    for i, config in enumerate(configs, 1):
        result = cross_validate(name, config, X_val, y_val, folds, n_classes, args)
        rows.append({**config, **result})
        print(f"[tune] {i:>3}/{len(configs)} {config} -> logloss {result['cv_logloss']:.4f} "
              f"acc {result['cv_accuracy']:.4f}", flush=True)
    rows.sort(key=lambda r: r["cv_logloss"])
    best = rows[0]
    best_config = {k: best[k] for k in GRIDS[name]}
    command = stacking_command(name, best_config, best.get("n_estimators"), args)

    print(f"\n[tune] done in {time.time() - start:.0f}s. Top 5 by CV log-loss:")
    for r in rows[:5]:
        print(f"  logloss {r['cv_logloss']:.4f} +/- {r['cv_logloss_std']:.4f}  "
              f"acc {r['cv_accuracy']:.4f}  "
              + ", ".join(f"{k}={r[k]}" for k in GRIDS[name])
              + (f", n_estimators={r['n_estimators']}" if "n_estimators" in r else ""))
    print(f"\n[tune] best configuration, to train on the whole val split:\n  {command}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"tune_{name}.json")
    with open(out_path, "w") as f:
        json.dump({"meta_learner": name, "folds": args.folds, "random_state": args.random_state,
                   "selection": "lowest mean CV log-loss", "grid": GRIDS[name],
                   "best": best, "command": command, "results": rows}, f, indent=2)
    print(f"[tune] saved -> {out_path}")


if __name__ == "__main__":
    main()
