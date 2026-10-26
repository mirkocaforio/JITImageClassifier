#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stacking ensemble (stacked generalization) of JITrans + RSCDMamba with a selectable
meta-learner: RandomForest (default), XGBoost, CatBoost or multinomial logistic regression
(see meta_learners.py).

Pipeline:
  1. Both base models export their softmax probabilities on the `val` and `test` splits
     (see export_probs.py / export_probs_mamba.py). This produces:
         <preds_dir>/jit_val.npz,  jit_test.npz
         <preds_dir>/mamba_val.npz, mamba_test.npz
  2. This script aligns the two models PER IMAGE using the rel_paths key, builds the
     meta-features X = concat(probs_jit, probs_mamba)  (N, 2*C), optionally as
     log-probabilities (--features logprobs), trains the meta-learner on the VAL split,
     and evaluates it on the TEST split.

No data leakage: the meta-learner is trained only on val predictions; test is held out
for the final evaluation. For the same reason the boosting models run a fixed number of
rounds: choose it, and the other hyper-parameters, with tune_meta.py (cross-validation
on val, early stopping inside the folds), never by looking at test.
Metrics are reported through the project's CrossEntropyLossStrategy.compute_metrics so
the format matches main.py. The meta-learner is also compared with Mamba, JITrans and the
plain average of their probabilities (McNemar's exact test, bootstrap 95% CI).

Outputs (in --out_dir): <meta_learner>_meta.joblib and stacking_test_metrics_<meta_learner>.json

Run in the JIT conda env:
    python stacking.py --preds_dir stacking/preds
    python stacking.py --meta_learner xgboost --learning_rate 0.05 --n_estimators 500
    python stacking.py --meta_learner catboost --max_depth 4
    python stacking.py --meta_learner logreg --features logprobs --C 0.1
    python stacking.py --out_dir ../datahome/staking_data/staking_no_classweight --class_weight none
"""

import argparse
import json
import os

import joblib
import numpy as np
import torch

from loss import CrossEntropyLossStrategy
from meta_learners import (
    CLASS_WEIGHTS,
    FEATURES,
    META_LEARNERS,
    build_meta_learner,
    check_labels,
    display_name,
    fit_meta_learner,
    meta_features,
    predict_proba_canonical,
)

# ---------------------------------------------------------------------------
# Loading / alignment
# ---------------------------------------------------------------------------

def load_preds(path):
    d = np.load(path, allow_pickle=True)
    return {
        "probs": d["probs"].astype(np.float64),
        "rel_paths": [str(p) for p in d["rel_paths"]],
        "targets": d["targets"].astype(np.int64),
        "classes": [str(c) for c in d["classes"]],
    }


def align_split(jit, mamba, split_name, min_overlap=0.5, return_rel_paths=False):
    """Align two models' predictions per image via rel_paths.

    Returns (X, y, classes) where X = concat(probs_jit, probs_mamba) row-aligned by image,
    and y is the (verified-consistent) ground-truth label.

    With return_rel_paths=True the tuple gains a fourth element: the rel_path of
    every kept row, in row order. inference.py uses it to map a prediction back
    to its image on disk (the rows are sorted by rel_path, so they do NOT follow
    the ImageFolder order).
    """
    if jit["classes"] != mamba["classes"]:
        raise ValueError(
            f"[{split_name}] class ordering differs between models:\n"
            f"  jit  : {jit['classes']}\n  mamba: {mamba['classes']}"
        )
    classes = jit["classes"]

    jit_map = {rp: (jit["probs"][i], jit["targets"][i]) for i, rp in enumerate(jit["rel_paths"])}
    mamba_map = {rp: (mamba["probs"][i], mamba["targets"][i]) for i, rp in enumerate(mamba["rel_paths"])}

    common = sorted(set(jit_map) & set(mamba_map))
    n_union = len(set(jit_map) | set(mamba_map))
    if len(common) == 0:
        raise ValueError(f"[{split_name}] no overlapping images between the two models.")
    overlap_frac = len(common) / max(n_union, 1)
    if overlap_frac < min_overlap:
        raise ValueError(
            f"[{split_name}] only {len(common)}/{n_union} images overlap "
            f"({overlap_frac:.1%}) — datasets/splits likely differ."
        )
    if len(common) < n_union:
        print(f"[stacking] WARNING [{split_name}] using {len(common)}/{n_union} "
              f"overlapping images ({overlap_frac:.1%}).")

    X_jit, X_mamba, y, kept_rel_paths = [], [], [], []
    n_target_mismatch = 0
    for rp in common:
        pj, tj = jit_map[rp]
        pm, tm = mamba_map[rp]
        if tj != tm:
            n_target_mismatch += 1
            continue
        X_jit.append(pj)
        X_mamba.append(pm)
        y.append(tj)
        kept_rel_paths.append(rp)
    if n_target_mismatch:
        print(f"[stacking] WARNING [{split_name}] dropped {n_target_mismatch} images "
              f"with mismatched ground-truth labels between models.")

    X = np.concatenate([np.asarray(X_jit), np.asarray(X_mamba)], axis=1)
    y = np.asarray(y, dtype=np.int64)
    print(f"[stacking] {split_name}: X={X.shape}, y={y.shape}")
    if return_rel_paths:
        return X, y, classes, kept_rel_paths
    return X, y, classes


# ---------------------------------------------------------------------------
# Metrics (reuse the project's reporting via the log-prob trick)
# ---------------------------------------------------------------------------

def metrics_from_probs(probs, targets, classes, detailed=True):
    """Run probs through CrossEntropyLossStrategy.compute_metrics.

    softmax(log(p)) == p, so feeding log-probabilities reproduces the original
    probabilities inside compute_metrics and reuses the exact project metric set.
    """
    strategy = CrossEntropyLossStrategy(num_classes=len(classes), class_names=classes,
                                        top_k=(1, 5) if len(classes) >= 5 else (1,))
    logits = torch.log(torch.from_numpy(np.asarray(probs, dtype=np.float64)) + 1e-12)
    return strategy.compute_metrics(logits, torch.from_numpy(np.asarray(targets)), detailed=detailed)


def base_model_probs_from_X(X, n_classes, which):
    """Slice the per-model probability block out of the stacked feature matrix."""
    if which == "jit":
        return X[:, :n_classes]
    return X[:, n_classes:2 * n_classes]


def paired_comparison(proba_a, proba_b, targets, n_boot=2000, seed=0):
    """Paired test of model a against model b on the same samples.

    McNemar's exact test (binomial on the discordant pairs) and a bootstrap 95% CI of the
    accuracy difference a - b. With accuracies this close the bare difference is not
    enough: on ~50k test images a few tens of images are within the noise.
    """
    from scipy.stats import binomtest

    correct_a = np.asarray(proba_a).argmax(1) == targets
    correct_b = np.asarray(proba_b).argmax(1) == targets
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    p_value = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0

    diff = correct_a.astype(np.float64) - correct_b
    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    return {
        "delta_accuracy": float(diff.mean()),
        "delta_accuracy_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "mcnemar_p": float(p_value),
    }


def print_comparisons(name, comparisons):
    print("\n[stacking] paired comparisons on test (McNemar exact, bootstrap 95% CI):")
    for other, c in comparisons.items():
        lo, hi = c["delta_accuracy_ci95"]
        print(f"  {name} vs {other:<10}: {c['delta_accuracy'] * 100:+.2f} pp "
              f"[{lo * 100:+.2f}, {hi * 100:+.2f}]  "
              f"({c['only_a_correct']} vs {c['only_b_correct']} discordant, "
              f"p = {c['mcnemar_p']:.2g})")


def print_report(title, metrics, classes):
    W = 62
    print("\n" + "=" * W)
    print(f"  {title}")
    print("=" * W)
    for k in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
              "precision_macro", "recall_macro", "cohen_kappa", "matthews_corrcoef"]:
        if k in metrics:
            print(f"  {k:<22}: {metrics[k]:.4f}")
    if "per_class" in metrics:
        print(f"  {'-' * (W - 2)}")
        print(f"  {'Class':<20}  Prec   Rec    F1     Support")
        for name, m in metrics["per_class"].items():
            print(f"  {name:<20}  {m['precision']:.3f}  {m['recall']:.3f}  "
                  f"{m['f1']:.3f}  {m['support']:>7}")
    if "confusion_matrix" in metrics:
        cm = np.asarray(metrics["confusion_matrix"])
        print(f"  {'-' * (W - 2)}")
        print("  Confusion matrix (rows=true, cols=pred):")
        print("    " + "".join(f"{i:>8d}" for i in range(cm.shape[1])))
        for i, row in enumerate(cm):
            print(f"    {i:<4d}" + "".join(f"{int(v):>8d}" for v in row))
        print("  Legend: " + ", ".join(f"{i}:{c}" for i, c in enumerate(classes)))
    print("=" * W)


def to_serializable(metrics):
    out = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = to_serializable(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser("Stacking of JITrans + Mamba with a selectable meta-learner")
    p.add_argument("--preds_dir", default="../datahome/stacking_data/preds", type=str)
    p.add_argument("--out_dir", default="../datahome/stacking_data/", type=str)
    p.add_argument("--meta_learner", default="rf", choices=META_LEARNERS,
                   help="Meta-learner trained on the base models' probabilities.")
    p.add_argument("--features", default="probs", choices=FEATURES,
                   help="Meta-features: the probabilities or their logarithms (suits logreg; "
                        "the trees are invariant to it).")
    # Hyper-parameters shared by all meta-learners
    p.add_argument("--class_weight", default="balanced", choices=CLASS_WEIGHTS,
                   help="'balanced_subsample' is available only for rf.")
    p.add_argument("--random_state", default=42, type=int)
    p.add_argument("--n_jobs", default=-1, type=int,
                   help="Threads of the tree ensembles (ignored by logreg).")
    # Tree ensembles only (rf / xgboost / catboost)
    p.add_argument("--n_estimators", default=300, type=int,
                   help="Trees of the forest (rf) or boosting rounds (xgboost / catboost).")
    p.add_argument("--max_depth", default=None, type=int,
                   help="Tree depth. Default: unlimited for rf, 6 for xgboost / catboost.")
    # Random Forest only
    p.add_argument("--min_samples_leaf", default=1, type=int)
    # Boosting only (xgboost / catboost)
    p.add_argument("--learning_rate", default=0.1, type=float)
    # XGBoost only
    p.add_argument("--subsample", default=1.0, type=float)
    p.add_argument("--colsample_bytree", default=1.0, type=float)
    p.add_argument("--min_child_weight", default=1.0, type=float)
    # CatBoost only
    p.add_argument("--l2_leaf_reg", default=3.0, type=float)
    # Logistic regression only
    p.add_argument("--C", default=1.0, type=float,
                   help="Inverse of the L2 regularisation strength.")
    p.add_argument("--max_iter", default=1000, type=int,
                   help="lbfgs iterations.")
    return p.parse_args()


def main():
    args = get_args()
    meta_name = display_name(args.meta_learner)

    jit_val = load_preds(os.path.join(args.preds_dir, "jit_val.npz"))
    jit_test = load_preds(os.path.join(args.preds_dir, "jit_test.npz"))
    mamba_val = load_preds(os.path.join(args.preds_dir, "mamba_val.npz"))
    mamba_test = load_preds(os.path.join(args.preds_dir, "mamba_test.npz"))

    X_val, y_val, classes = align_split(jit_val, mamba_val, "val")
    X_test, y_test, _ = align_split(jit_test, mamba_test, "test")
    n_classes = len(classes)
    check_labels(y_val, n_classes, "val")

    # --- Train the meta-learner on the val predictions ---
    meta = build_meta_learner(
        args.meta_learner,
        n_classes,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_samples_leaf=args.min_samples_leaf,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        l2_leaf_reg=args.l2_leaf_reg,
        C=args.C,
        max_iter=args.max_iter,
        class_weight=args.class_weight,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    print(f"[stacking] training {meta_name} on val ({args.features}): {X_val.shape} ...")
    fit_meta_learner(meta, meta_features(X_val, args.features), y_val,
                     class_weight=args.class_weight)

    # --- Predict on test, columns in the canonical 0..C-1 class order ---
    proba = predict_proba_canonical(meta, meta_features(X_test, args.features), n_classes)

    # --- Reporting: base models alone vs. the stacked meta-learner ---
    jit_proba = base_model_probs_from_X(X_test, n_classes, "jit")
    mamba_proba = base_model_probs_from_X(X_test, n_classes, "mamba")
    # Untrained baseline: plain average of the two models' probabilities
    avg_proba = (jit_proba + mamba_proba) / 2
    jit_test_metrics = metrics_from_probs(jit_proba, y_test, classes, detailed=True)
    mamba_test_metrics = metrics_from_probs(mamba_proba, y_test, classes, detailed=True)
    avg_test_metrics = metrics_from_probs(avg_proba, y_test, classes, detailed=True)
    meta_test_metrics = metrics_from_probs(proba, y_test, classes, detailed=True)

    print_report("BASE — JITrans (test)", jit_test_metrics, classes)
    print_report("BASE — Mamba (test)", mamba_test_metrics, classes)
    print_report("BASELINE — average of the probabilities (test)", avg_test_metrics, classes)
    print_report(f"STACKING — {meta_name} (test)", meta_test_metrics, classes)

    comparisons = {other: paired_comparison(proba, other_proba, y_test, seed=args.random_state)
                   for other, other_proba in (("mamba", mamba_proba), ("jit", jit_proba),
                                              ("average", avg_proba))}
    print_comparisons(meta_name, comparisons)

    best_base = max(jit_test_metrics["accuracy"], mamba_test_metrics["accuracy"])
    print(f"\n[stacking] test accuracy — JITrans: {jit_test_metrics['accuracy']:.4f} | "
          f"Mamba: {mamba_test_metrics['accuracy']:.4f} | "
          f"{meta_name} stacking: {meta_test_metrics['accuracy']:.4f}")
    delta = meta_test_metrics["accuracy"] - best_base
    print(f"[stacking] {meta_name} vs best base: {delta:+.4f} accuracy")

    # --- Persist artefacts ---
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, f"{args.meta_learner}_meta.joblib")
    joblib.dump({"meta_learner": args.meta_learner, "model": meta, "classes": classes,
                 "feature_order": ["jit", "mamba"], "features": args.features}, model_path)

    metrics_path = os.path.join(args.out_dir,
                                f"stacking_test_metrics_{args.meta_learner}.json")
    payload = {
        "classes": classes,
        "meta_learner": args.meta_learner,
        "features": args.features,
        "meta_params": meta.get_params(),
        "base_jit_test": to_serializable(jit_test_metrics),
        "base_mamba_test": to_serializable(mamba_test_metrics),
        "baseline_average_test": to_serializable(avg_test_metrics),
        f"stacking_{args.meta_learner}_test": to_serializable(meta_test_metrics),
        f"stacking_{args.meta_learner}_vs": comparisons,
    }
    with open(metrics_path, "w") as f:
        # default=str: some library parameters (e.g. callables) are not JSON types
        json.dump(payload, f, indent=2, default=str)

    print(f"\n[stacking] saved meta-model  -> {model_path}")
    print(f"[stacking] saved metrics     -> {metrics_path}")


if __name__ == "__main__":
    main()
