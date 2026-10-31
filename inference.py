#!/usr/bin/env python3

"""
Inference, evaluation and reporting for the JITrans classifier and for the
stacking meta-learner (RandomForest, XGBoost, CatBoost or multinomial
logistic regression).

This is the JIT-side twin of `rscd_mamba/core/processing/inference.py`: same
metric set, same plots, same exported files, so that JITrans, RSCDMamba and the
stacking ensemble can be compared one to one.

It is deliberately **self-contained**: this repo has no `core/` package, so
nothing here imports `core.*`, `CheckpointLoader` or `RSCDMamba`.  The only
local dependencies are `dataloader.build_dataset`, `model.define_model`,
and (for the stacking path) `stacking.load_preds` / `align_split` and
`meta_learners`.

Three entry points, selected by `--model` (or by `--train-log` alone):

1. `JITLogParser` / `TrainingHistory`
   Rebuild the per-epoch history from `output/<run>/train.log` written by
   `main.py` and plot loss / accuracy / F1 / confidence curves.

2. `JITInference`
   Loads a JITrans checkpoint, runs the forward pass over the RSCD val/test
   splits and reports the full extended metric set, plus optional
   complexity / latency / memory benchmarks.

3. `StackingInference`
   Loads `stacking/<rf|xgboost|catboost|logreg>_meta.joblib` (written by stacking.py)
   plus the exported `.npz` probability files and evaluates the meta-learner
   through the very same reporting layer.  `--compare` puts JITrans, Mamba and
   the stacking side by side.  Exported files are named after
   `stacking_<meta-learner>`, so runs with different meta-learners can share
   one --export-dir.

Everything is saved as **PDF** by default (vector, publication ready); PNG is
available via `--plot-format`.

Examples
--------
    # training curves only (no model, no dataset needed)
    python inference.py --train-log output/jitrans_20260615_155632_1gpu/train.log \
        --plot-dir ./eval --export-dir ./eval

    # full JITrans evaluation on val + test, reported independently
    python inference.py --model jit \
        --resume output/jitrans_20260615_155632_1gpu/best.pt \
        --data-path ../RSCD_dataset-1million_reorganized/ \
        --split separate --plot-all --export-dir ./eval

    # stacking meta-learner, with the three-way model comparison
    python inference.py --model stacking \
        --meta-model stacking/rf_meta.joblib --preds-dir stacking/preds \
        --split test --plot-all --compare --export-dir ./eval

    # complexity + latency, saved as model_summary_jit.* / single_inference_jit.*
    python inference.py --model jit --resume .../best.pt --data-path ... \
        --split test --model-summary --benchmark --memory --inference \
        --export-dir ./eval

    # stacking: meta-learner complexity/latency + full-pipeline figures composed
    # from model_summary_{jit,mamba}.json and single_inference_{jit,mamba}.json
    # (found in --export-dir / --preds-dir, or passed with --base-files)
    python inference.py --model stacking --meta-model stacking/xgboost_meta.joblib \
        --preds-dir stacking/preds --split test --model-summary --benchmark \
        --inference --export-dir ./eval
"""

from __future__ import annotations

from typing import Optional, Union, Dict, List, Tuple, Any, Sequence
import argparse
import contextlib
import csv
import datetime as _dt
import glob as globmod
import json
import math
import os
import re
import sys
import time

import numpy as np

# Select a non-interactive backend before pyplot is imported: training and
# evaluation normally run on a headless server.
import matplotlib
if (os.environ.get('MPLBACKEND') is None
        and sys.platform.startswith('linux')
        and not os.environ.get('DISPLAY')):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# torch and scikit-learn are needed to *evaluate* a model, but not to parse a
# training log and draw its curves.  Importing them lazily keeps `--train-log`
# usable from a plain numpy+matplotlib environment, which is exactly what its
# help text promises.
_MISSING_DEPS: List[str] = []

try:
    import torch
    from torch.utils.data import DataLoader, ConcatDataset
except ImportError:
    torch = None
    DataLoader = ConcatDataset = None
    _MISSING_DEPS.append('torch')

FLOP_COUNTER_AVAILABLE = False
if torch is not None:
    try:
        from torch.utils.flop_counter import FlopCounterMode
        FLOP_COUNTER_AVAILABLE = True
    except ImportError:
        pass

try:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        precision_score, recall_score, f1_score,
        confusion_matrix, classification_report,
        cohen_kappa_score, matthews_corrcoef, top_k_accuracy_score,
        roc_curve, auc,
        precision_recall_curve, average_precision_score
    )
    from sklearn.preprocessing import label_binarize
except ImportError:
    _MISSING_DEPS.append('scikit-learn')


def require_eval_deps(*needed: str):
    """
    Fail with an actionable message instead of a bare ImportError.

    Called with no arguments it requires everything; the stacking path asks
    only for scikit-learn, since the meta-learner needs no torch at all.
    """
    wanted = set(needed) if needed else {'torch', 'scikit-learn'}
    missing = [d for d in _MISSING_DEPS if d in wanted]
    if missing:
        raise ImportError(
            f"{', '.join(missing)} is required for this action. "
            "Run it in the JIT conda env (see requirements.txt). "
            "--train-log works without any of it."
        )


# ======================================================================
# Console helpers
# ======================================================================

# The Windows console defaults to cp1252 and raises UnicodeEncodeError on the
# box-drawing / arrow glyphs used by the rscd_mamba script, so every marker
# here stays ASCII.
OK   = '[ok]'
WARN = '[!]'
ARROW = '->'


# ======================================================================
# Figure saving helper
# ======================================================================

PLOT_FORMATS = ('pdf', 'png', 'both')


def save_figure(
    fig,
    save_dir: Optional[str],
    basename: str,
    plot_format: str = 'pdf',
    dpi: int = 150,
    show: bool = False,
    verbose: bool = True,
) -> List[str]:
    """
    Save a matplotlib figure as PDF and/or PNG.

    Args:
        fig:         The figure to save.
        save_dir:    Destination directory (created if missing).  None -> not saved.
        basename:    File name without extension.
        plot_format: 'pdf' (default), 'png' or 'both'.
        dpi:         Raster resolution - only relevant for PNG.
        show:        Call plt.show() before closing the figure.
        verbose:     Print the written paths.

    Returns:
        List of written file paths.
    """
    if plot_format not in PLOT_FORMATS:
        raise ValueError(f"plot_format must be one of {PLOT_FORMATS}, got '{plot_format}'")

    paths: List[str] = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        formats = ['pdf', 'png'] if plot_format == 'both' else [plot_format]
        for fmt in formats:
            path = os.path.join(save_dir, f"{basename}.{fmt}")
            fig.savefig(path, format=fmt, dpi=dpi, bbox_inches='tight')
            paths.append(path)
            if verbose:
                print(f"  {OK} saved {ARROW} {path}")

    if show:
        plt.show()
    plt.close(fig)
    return paths


# ======================================================================
# ROC / Precision-Recall curve helpers
# ======================================================================

# Line styles of the averaged curves, shared by every curve figure
MICRO_STYLE = dict(color='deeppink', lw=2.5, linestyle=':')
MACRO_STYLE = dict(color='black', lw=2.5, linestyle='--')


def compute_curve_data(
    targets: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> Optional[Dict[str, Any]]:
    """
    One-vs-rest ROC / PR curves of every class plus the micro and macro averages.

    A class with no positive (or no negative) sample in the split has no
    defined curve: it is listed in 'skipped' and left out of the averages,
    which would otherwise turn NaN.

    Returns None when no class has a defined curve, otherwise a dict with:
        'roc'      {class index: (fpr, tpr, auc)}
        'pr'       {class index: (recall, precision, average precision)}
        'micro'    {'roc': (fpr, tpr, auc), 'pr': (recall, precision, ap)}
                   every (sample, class) pair pooled into one binary problem
        'macro'    same layout, plus the 'roc_weighted' / 'pr_weighted' scores
                   (per-class scores weighted by support)
        'valid', 'skipped'  class indices
        'support'  number of samples of each class
    'micro' and 'macro' are None with two classes: the two columns are
    complementary, so the positive-class curve already says everything.
    """
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities, dtype=float)
    # label_binarize would collapse to a single column with two classes
    y_true = np.eye(num_classes, dtype=int)[targets]
    support = y_true.sum(axis=0)
    valid = [i for i in range(num_classes) if 0 < support[i] < len(targets)]
    skipped = [i for i in range(num_classes) if i not in valid]
    if not valid:
        return None

    roc: Dict[int, Tuple] = {}
    pr: Dict[int, Tuple] = {}
    for i in valid:
        fpr, tpr, _ = roc_curve(y_true[:, i], probabilities[:, i])
        roc[i] = (fpr, tpr, float(auc(fpr, tpr)))
        precision, recall, _ = precision_recall_curve(y_true[:, i], probabilities[:, i])
        pr[i] = (recall, precision,
                 float(average_precision_score(y_true[:, i], probabilities[:, i])))

    data: Dict[str, Any] = {'roc': roc, 'pr': pr, 'micro': None, 'macro': None,
                            'valid': valid, 'skipped': skipped, 'support': support}
    if num_classes == 2:
        return data

    y_flat, p_flat = y_true.ravel(), probabilities.ravel()
    fpr, tpr, _ = roc_curve(y_flat, p_flat)
    precision, recall, _ = precision_recall_curve(y_flat, p_flat)
    data['micro'] = {
        'roc': (fpr, tpr, float(auc(fpr, tpr))),
        'pr': (recall, precision, float(average_precision_score(y_flat, p_flat))),
    }

    # Macro ROC: mean TPR on the union of every class's FPR points
    all_fpr = np.unique(np.concatenate([roc[i][0] for i in valid]))
    mean_tpr = np.mean([np.interp(all_fpr, roc[i][0], roc[i][1]) for i in valid], axis=0)
    # Macro PR: mean precision on a common recall grid; its score is the
    # mean of the per-class APs (the area under an averaged PR curve is not)
    recall_grid = np.linspace(0.0, 1.0, 1001)
    mean_precision = np.mean([np.interp(recall_grid, pr[i][0][::-1], pr[i][1][::-1])
                              for i in valid], axis=0)
    weights = support[valid] / support[valid].sum()
    data['macro'] = {
        'roc': (all_fpr, mean_tpr, float(auc(all_fpr, mean_tpr))),
        'pr': (recall_grid, mean_precision, float(np.mean([pr[i][2] for i in valid]))),
        'roc_weighted': float(np.sum(weights * np.array([roc[i][2] for i in valid]))),
        'pr_weighted': float(np.sum(weights * np.array([pr[i][2] for i in valid]))),
    }
    return data


def auc_scores_from_curves(curves: Dict[str, Any], class_names: List[str]) -> Dict[str, float]:
    """
    Flatten compute_curve_data() into the 'auc_scores' dict:
    roc_auc_<class>, pr_auc_<class> and roc_auc_/pr_auc_ macro, micro, weighted.
    With two classes the averages are the positive-class (index 1) scores.
    """
    scores: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        scores[f'roc_auc_{name}'] = curves['roc'][i][2] if i in curves['roc'] else float('nan')
        scores[f'pr_auc_{name}'] = curves['pr'][i][2] if i in curves['pr'] else float('nan')

    if curves['macro'] is None:
        positive = 1 if 1 in curves['roc'] else curves['valid'][0]
        for average in ('macro', 'micro', 'weighted'):
            scores[f'roc_auc_{average}'] = curves['roc'][positive][2]
            scores[f'pr_auc_{average}'] = curves['pr'][positive][2]
        return scores

    for kind in ('roc', 'pr'):
        scores[f'{kind}_auc_macro'] = curves['macro'][kind][2]
        scores[f'{kind}_auc_micro'] = curves['micro'][kind][2]
        scores[f'{kind}_auc_weighted'] = curves['macro'][f'{kind}_weighted']
    return scores


def _value_at(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """np.interp that also accepts a decreasing `xs` (precision_recall_curve)."""
    if xs[0] > xs[-1]:
        xs, ys = xs[::-1], ys[::-1]
    return float(np.interp(x, xs, ys))


def _add_zoom_inset(ax, lines, xlim, ylim, bounds) -> None:
    """
    Magnify a corner of `ax` in an inset.  `lines` is a list of
    (x, y, plot kwargs) already drawn on `ax`; `ylim[0]` is rounded down to
    a multiple of 0.05.
    """
    inset = ax.inset_axes(bounds)
    for x, y, style in lines:
        inset.plot(x, y, **style)
    inset.set_xlim(*xlim)
    inset.set_ylim(max(0.0, np.floor(min(ylim[0], 0.9) * 20) / 20), ylim[1])
    inset.tick_params(labelsize=8)
    inset.grid(True, alpha=0.3)
    ax.indicate_inset_zoom(inset, edgecolor='0.4')


def _draw_iso_f1(ax, levels=(0.2, 0.4, 0.6, 0.8)) -> None:
    """Grey iso-F1 curves on a precision-recall axes."""
    for f1 in levels:
        recall = np.linspace(f1 / (2 - f1) + 1e-3, 1, 200)
        precision = f1 * recall / (2 * recall - f1)
        ax.plot(recall, precision, color='gray', alpha=0.3, lw=1)
        ax.annotate(f"F1={f1:.1f}", xy=(0.88, precision[-1] + 0.02),
                    fontsize=8, color='gray')
    ax.plot([], [], color='gray', alpha=0.3, lw=1, label="iso-F1 curves")


# ======================================================================
# Training log parsing
# ======================================================================

# main.py logs through logging.Formatter("%(asctime)s  %(message)s"), so every
# line starts with "YYYY-mm-dd HH:MM:SS" followed by two spaces.
_TS_RE = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})  (.*)$')

# main.py:452 - the compact one-liner, written for EVERY epoch.
_EPOCH_RE = re.compile(
    r'^Epoch\s+(\d+)/(\d+)\s*\|\s*'
    r'Train loss:\s*([\d.eE+-]+)\s*\|\s*'
    r'Val loss:\s*([\d.eE+-]+)\s+'
    r'acc:\s*([\d.eE+-]+)\s+'
    r'f1:\s*([\d.eE+-]+)'
)

# main.py:295 - opens the detailed block, emitted only on best epochs.
_BEST_HEADER_RE = re.compile(r'^\s*NEW BEST\s+\S+\s+Epoch\s+(\d+)\s*$')
# main.py:314 - the end-of-training test block; must NOT land on an epoch.
_TEST_HEADER_RE = re.compile(r'^\s*TEST\b.*FINAL RESULTS\s*$')
# main.py:218-246 - "  <key><padding>: <value>" inside either block.
_BLOCK_METRIC_RE = re.compile(r'^\s{2,}([A-Za-z][A-Za-z0-9_]*)\s*:\s*([-\d.eE+]+|nan)\s*$')
# "Val Loss" / "Test Loss" carry a space, so they need their own pattern.
_BLOCK_LOSS_RE = re.compile(r'^\s{2,}(Val|Test) Loss\s*:\s*([-\d.eE+]+|nan)\s*$')

_HEADER_PATTERNS = {
    'device':      re.compile(r'^Using device:\s*(.+)$'),
    'log_file':    re.compile(r'^Log file\s*:\s*(.+)$'),
    'classes':     re.compile(r'^Classes\s*\(\d+\):\s*(.+)$'),
    'split_sizes': re.compile(r'^(Train:\s*\d+\s*\|\s*Val:.*)$'),
    'resumed_from': re.compile(r'^Resuming from checkpoint:\s*(.+)$'),
    'best_checkpoint': re.compile(r'^\s*Best model saved to\s+(\S+)'),
}

# Metrics that only appear inside a best/test block, stored as `val_<key>`.
_BLOCK_KEYS = {
    'accuracy', 'balanced_accuracy', 'top_1_accuracy', 'top_5_accuracy',
    'f1_macro', 'f1_weighted', 'f1_micro',
    'precision_macro', 'precision_weighted', 'recall_macro', 'recall_weighted',
    'cohen_kappa', 'matthews_corrcoef',
    'mean_confidence', 'median_confidence', 'min_confidence',
    'mean_correct_confidence', 'mean_incorrect_confidence',
}


class TrainingHistory:
    """
    Per-epoch training history reconstructed from one or more training logs.

    Attributes:
        epochs:  1-D int array of epoch numbers (1-based, sorted, may have gaps).
        metrics: Dict metric-name -> float array aligned with `epochs`.
                 Missing values are NaN, so matplotlib naturally breaks the line.
        meta:    Header information (device, classes, split sizes, final test
                 metrics as `test_<key>`, ...).
        files:   Log files that contributed at least one epoch.
        summary: Derived facts (best epochs, gaps, total wall time).
    """

    def __init__(
        self,
        epochs: np.ndarray,
        metrics: Dict[str, np.ndarray],
        meta: Dict[str, str],
        files: List[str],
        skipped_files: List[str],
    ):
        self.epochs = epochs
        self.metrics = metrics
        self.meta = meta
        self.files = files
        self.skipped_files = skipped_files
        self.summary = self._build_summary()

    # -- access -------------------------------------------------------

    def get(self, key: str) -> Optional[np.ndarray]:
        """Return the array for `key`, or None if absent / entirely NaN."""
        arr = self.metrics.get(key)
        if arr is None or np.all(np.isnan(arr)):
            return None
        return arr

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    @property
    def available_metrics(self) -> List[str]:
        return sorted(k for k in self.metrics if self.has(k))

    def __len__(self) -> int:
        return len(self.epochs)

    # -- summary ------------------------------------------------------

    def _best(self, key: str, mode: str) -> Optional[Tuple[int, float]]:
        arr = self.get(key)
        if arr is None:
            return None
        idx = int(np.nanargmin(arr) if mode == 'min' else np.nanargmax(arr))
        return int(self.epochs[idx]), float(arr[idx])

    def _build_summary(self) -> Dict[str, Any]:
        expected = set(range(int(self.epochs[0]), int(self.epochs[-1]) + 1)) if len(self.epochs) else set()
        gaps = sorted(expected - set(int(e) for e in self.epochs))

        epoch_time = self.get('epoch_time_s')
        total_hours = float(np.nansum(epoch_time)) / 3600.0 if epoch_time is not None else None

        return {
            'n_epochs': len(self.epochs),
            'first_epoch': int(self.epochs[0]) if len(self.epochs) else None,
            'last_epoch': int(self.epochs[-1]) if len(self.epochs) else None,
            'missing_epochs': gaps,
            'total_train_hours': total_hours,
            'best_val_loss': self._best('val_loss', 'min'),
            'best_train_loss': self._best('train_loss', 'min'),
            'best_val_accuracy': self._best('val_accuracy', 'max'),
            'best_val_balanced_accuracy': self._best('val_balanced_accuracy', 'max'),
            'best_val_f1_macro': self._best('val_f1_macro', 'max'),
        }

    def print_summary(self):
        s = self.summary
        print(f"\n{'='*66}")
        print("TRAINING LOG SUMMARY")
        print(f"{'='*66}")
        for path in self.files:
            print(f"  log: {path}")
        for path in self.skipped_files:
            print(f"  skipped (no epochs): {path}")
        if self.meta:
            print(f"  {'-'*62}")
            for key in ('device', 'classes', 'split_sizes', 'total_epochs',
                        'resumed_from', 'best_checkpoint'):
                if key in self.meta:
                    value = self.meta[key]
                    if len(value) > 100:
                        value = value[:97] + '...'
                    print(f"  {key:<16} {value}")
        print(f"  {'-'*62}")
        print(f"  Epochs parsed:   {s['n_epochs']}  (from {s['first_epoch']} to {s['last_epoch']})")
        if s['missing_epochs']:
            shown = s['missing_epochs'][:15]
            more = '' if len(s['missing_epochs']) == len(shown) else f" (+{len(s['missing_epochs'])-len(shown)} more)"
            print(f"  Missing epochs:  {shown}{more}")
        if s['total_train_hours'] is not None:
            print(f"  Total wall time: {s['total_train_hours']:.2f} h  (from log timestamps)")
        for label, key in (('Best train loss', 'best_train_loss'),
                           ('Best val loss', 'best_val_loss'),
                           ('Best val accuracy', 'best_val_accuracy'),
                           ('Best val bal. acc.', 'best_val_balanced_accuracy'),
                           ('Best val F1 macro', 'best_val_f1_macro')):
            if s[key] is not None:
                epoch, value = s[key]
                print(f"  {label:<20} {value:.6f}  @ epoch {epoch}")

        test_keys = sorted(k for k in self.meta if k.startswith('test_'))
        if test_keys:
            print(f"  {'-'*62}")
            print("  Final test block found in the log:")
            for key in test_keys:
                print(f"    {key[5:]:<26} {self.meta[key]}")
        print(f"  Metrics found:   {len(self.available_metrics)}")
        print(f"{'='*66}")

    # -- export -------------------------------------------------------

    def to_csv(self, path: str) -> str:
        """Dump the whole per-epoch history to a CSV file."""
        keys = self.available_metrics
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['epoch'] + keys)
            for i, epoch in enumerate(self.epochs):
                row = [int(epoch)]
                for key in keys:
                    value = self.metrics[key][i]
                    row.append('' if np.isnan(value) else f"{value:.12g}")
                writer.writerow(row)
        print(f"  {OK} saved {ARROW} {path}")
        return path


class JITLogParser:
    """
    Parse the plain-text training logs produced by `main.py`.

    Log shape::

        2026-06-17 07:13:08  Epoch 51/150 | Train loss: 0.6632 | Val loss: 0.5604  acc: 0.9303  f1: 0.9489
        2026-06-17 07:28:27  Epoch 52/150 | Train loss: 0.6620 | Val loss: 0.5566  acc: 0.9321  f1: 0.9519  * new best
        2026-06-17 07:28:34    NEW BEST  -  Epoch 52
        2026-06-17 07:28:34    Val Loss                          : 0.5566
        2026-06-17 07:28:34    accuracy                          : 0.9321
        2026-06-17 07:28:34    balanced_accuracy                 : 0.9529
        ...

    Notes that drive the implementation:
      * The compact one-liner exists for every epoch and carries train_loss,
        val_loss, val_accuracy and val_f1_macro.
      * The detailed `NEW BEST` block is emitted only on best epochs
        (main.py:464-477), so the richer metrics are sparse - stored as NaN
        elsewhere, which makes matplotlib break the line by itself.
      * The trailing `TEST - FINAL RESULTS` block belongs to no epoch; it is
        parsed into `meta['test_<key>']` instead of polluting the curves.
      * `main.py` never logs the learning rate, so the LR plot is skipped.
      * `epoch_time_s` is not logged either; it is derived from the delta
        between consecutive epoch-line timestamps (first epoch stays NaN).
    """

    LOG_GLOB = 'train*.log'

    # -- path resolution ---------------------------------------------

    @classmethod
    def resolve_paths(cls, paths: Union[str, Sequence[str]], verbose: bool = True) -> List[str]:
        """Expand files, globs and directories into a sorted list of log files."""
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]

        resolved: List[str] = []
        for raw in paths:
            path = str(raw)
            if os.path.isdir(path):
                found = sorted(globmod.glob(os.path.join(path, cls.LOG_GLOB)))
                if not found:
                    found = sorted(globmod.glob(os.path.join(path, '*.log')))
                if not found and verbose:
                    print(f"{WARN} No log files in directory: {path}")
                resolved.extend(found)
            elif any(ch in path for ch in '*?['):
                found = sorted(globmod.glob(path))
                if not found and verbose:
                    print(f"{WARN} Glob matched nothing: {path}")
                resolved.extend(found)
            elif os.path.isfile(path):
                resolved.append(path)
            elif verbose:
                print(f"{WARN} Training log not found, skipped: {path}")

        # De-duplicate while keeping order
        seen, unique = set(), []
        for path in resolved:
            key = os.path.abspath(path)
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    # -- single file --------------------------------------------------

    @classmethod
    def _parse_file(cls, path: str) -> Tuple[Dict[int, Dict[str, float]], Dict[str, str]]:
        epochs: Dict[int, Dict[str, float]] = {}
        meta: Dict[str, str] = {}
        # 'best' -> writing into epochs[block_epoch]; 'test' -> writing into meta
        block_mode: Optional[str] = None
        block_epoch: Optional[int] = None
        previous_ts: Optional[_dt.datetime] = None

        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            for raw_line in fh:
                match = _TS_RE.match(raw_line.rstrip('\n'))
                if not match:
                    # Continuation lines without a timestamp (tqdm noise,
                    # tracebacks) are ignored on purpose.
                    continue
                timestamp_str, body = match.group(1), match.group(2)

                # 1) the compact per-epoch line
                epoch_match = _EPOCH_RE.match(body.strip())
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    record = epochs.setdefault(epoch, {})
                    record['train_loss'] = float(epoch_match.group(3))
                    record['val_loss'] = float(epoch_match.group(4))
                    record['val_accuracy'] = float(epoch_match.group(5))
                    record['val_f1_macro'] = float(epoch_match.group(6))

                    meta['total_epochs'] = epoch_match.group(2)

                    # Wall time is not logged; derive it from the timestamps.
                    try:
                        current_ts = _dt.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        if previous_ts is not None:
                            delta = (current_ts - previous_ts).total_seconds()
                            # A resumed run restarts the clock; ignore negatives
                            # and implausible gaps between separate sessions.
                            if 0 < delta < 24 * 3600:
                                record['epoch_time_s'] = float(delta)
                        previous_ts = current_ts
                    except ValueError:
                        pass

                    block_mode, block_epoch = None, None
                    continue

                # 2) block headers
                best_match = _BEST_HEADER_RE.match(body)
                if best_match:
                    block_mode, block_epoch = 'best', int(best_match.group(1))
                    continue
                if _TEST_HEADER_RE.match(body):
                    block_mode, block_epoch = 'test', None
                    continue

                # 3) "  key: value" lines inside a block
                if block_mode is not None:
                    loss_match = _BLOCK_LOSS_RE.match(body)
                    if loss_match:
                        value = float(loss_match.group(2))
                        if block_mode == 'test':
                            meta['test_loss'] = f"{value:.4f}"
                        elif block_epoch is not None:
                            epochs.setdefault(block_epoch, {})['val_loss'] = value
                        continue

                    metric_match = _BLOCK_METRIC_RE.match(body)
                    if metric_match:
                        key, raw_value = metric_match.group(1), metric_match.group(2)
                        if key in _BLOCK_KEYS:
                            try:
                                value = float(raw_value)
                            except ValueError:
                                continue
                            if block_mode == 'test':
                                meta[f'test_{key}'] = f"{value:.4f}"
                            elif block_epoch is not None:
                                epochs.setdefault(block_epoch, {})[f'val_{key}'] = value
                        continue

                    # Per-class rows, confusion matrix and separators end the
                    # scalar section but not the block; just skip them.
                    continue

                # 4) header metadata outside any block
                for key, pattern in _HEADER_PATTERNS.items():
                    header_match = pattern.match(body)
                    if header_match:
                        meta[key] = header_match.group(1).strip()
                        break

        return epochs, meta

    # -- public entry point -------------------------------------------

    @classmethod
    def parse(
        cls,
        paths: Union[str, Sequence[str]],
        verbose: bool = True,
    ) -> TrainingHistory:
        """
        Parse one or more training logs into a `TrainingHistory`.

        Args:
            paths:   File, glob, directory, or list of any of those.
            verbose: Print per-file diagnostics.

        Returns:
            TrainingHistory with epochs merged across files.
        """
        files = cls.resolve_paths(paths, verbose=verbose)
        if not files:
            raise FileNotFoundError(f"No training log file resolved from: {paths}")

        merged: Dict[int, Dict[str, float]] = {}
        meta: Dict[str, str] = {}
        used_files: List[str] = []
        skipped_files: List[str] = []

        for path in files:
            try:
                epochs, file_meta = cls._parse_file(path)
            except OSError as exc:
                print(f"{WARN} Could not read {path}: {exc}")
                skipped_files.append(path)
                continue

            if not epochs:
                if verbose:
                    print(f"{WARN} No epoch records in {os.path.basename(path)} "
                          f"(aborted/crashed run?) - skipped")
                skipped_files.append(path)
                continue

            meta.update(file_meta)      # later file wins
            merged.update(epochs)       # later file wins on overlapping epochs
            used_files.append(path)
            if verbose:
                print(f"{OK} {os.path.basename(path)}: {len(epochs)} epochs "
                      f"({min(epochs)}-{max(epochs)})")

        if not merged:
            raise ValueError(
                f"None of the {len(files)} log file(s) contained a completed epoch. "
                "Training probably crashed before the first epoch ended."
            )

        epoch_numbers = sorted(merged)
        keys = sorted({key for record in merged.values() for key in record})
        metrics = {
            key: np.array([merged[epoch].get(key, np.nan) for epoch in epoch_numbers], dtype=float)
            for key in keys
        }

        history = TrainingHistory(
            epochs=np.array(epoch_numbers, dtype=int),
            metrics=metrics,
            meta=meta,
            files=used_files,
            skipped_files=skipped_files,
        )
        if verbose:
            history.print_summary()
        return history


# Backwards-friendly alias: same name as the rscd_mamba parser.
TrainingLogParser = JITLogParser


# ======================================================================
# Training-curve plots (no model / dataset required)
# ======================================================================

def _plot_series(ax, x, history: TrainingHistory, key: str, label: str, **kwargs) -> bool:
    """Plot history[key] on ax; return False if the metric is unavailable."""
    values = history.get(key)
    if values is None:
        return False
    ax.plot(x, values, label=label, **kwargs)
    return True


def _mark_best(ax, history: TrainingHistory, key: str, mode: str, color: str = 'crimson'):
    """Add a marker + annotation on the best epoch of `key`."""
    values = history.get(key)
    if values is None:
        return
    idx = int(np.nanargmin(values) if mode == 'min' else np.nanargmax(values))
    epoch, value = int(history.epochs[idx]), float(values[idx])
    ax.plot(epoch, value, marker='o', markersize=8, markerfacecolor='none',
            markeredgecolor=color, markeredgewidth=2, linestyle='none', zorder=5)
    ax.annotate(
        f"best: {value:.4f}\n@ epoch {epoch}",
        xy=(epoch, value),
        xytext=(8, 12 if mode == 'min' else -28),
        textcoords='offset points',
        fontsize=8, color=color,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color, alpha=0.85),
    )


def plot_training_curves(
    history: TrainingHistory,
    save_dir: Optional[str] = None,
    plot_format: str = 'pdf',
    show: bool = False,
    dpi: int = 150,
    log_scale: bool = False,
    prefix: str = 'training',
) -> Dict[str, List[str]]:
    """
    Plot the evolution of loss, accuracy, F1 and confidence across epochs,
    from a parsed training log.

    Produces (one file each):
        <prefix>_loss        train_loss vs val_loss
        <prefix>_accuracy    val accuracy / balanced accuracy / top-5
        <prefix>_f1          val F1 macro / weighted / micro
        <prefix>_lr          learning-rate schedule (skipped: not logged)
        <prefix>_confidence  mean / correct / incorrect confidence
        <prefix>_overview    2x2 summary panel

    Note: `main.py` only logs `train_loss` - there is no per-epoch train
    accuracy - so the accuracy plot shows validation metrics only.  Metrics
    that come from the `NEW BEST` block exist only on best epochs, hence the
    markers-and-gaps look of those series.

    Returns:
        Dict figure-name -> list of written paths.
    """
    x = history.epochs
    written: Dict[str, List[str]] = {}
    title_suffix = ''
    if 'log_file' in history.meta:
        run = os.path.basename(os.path.dirname(history.meta['log_file']))
        if run:
            title_suffix = f" - {run}"

    print(f"\n{'='*66}")
    print("TRAINING CURVES")
    print(f"{'='*66}")

    # -- Loss ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    has_train = _plot_series(ax, x, history, 'train_loss', 'Train loss',
                             color='#1f77b4', lw=1.8)
    has_val = _plot_series(ax, x, history, 'val_loss', 'Validation loss',
                           color='#d62728', lw=1.8)
    if has_train or has_val:
        _mark_best(ax, history, 'val_loss', 'min')
        if log_scale:
            ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss' + (' (log scale)' if log_scale else ''), fontsize=12)
        ax.set_title(f"Training and validation loss{title_suffix}",
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        written['loss'] = save_figure(fig, save_dir, f"{prefix}_loss",
                                      plot_format, dpi, show)
    else:
        plt.close(fig)
        print(f"  {WARN} no loss metrics in the log - loss plot skipped")

    # -- Accuracy -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    drawn = [
        _plot_series(ax, x, history, 'val_accuracy', 'Val accuracy',
                     color='#2ca02c', lw=1.8),
        _plot_series(ax, x, history, 'val_balanced_accuracy',
                     'Val balanced accuracy (best epochs only)',
                     color='#ff7f0e', lw=1.5, marker='o', ms=3),
        _plot_series(ax, x, history, 'val_top_5_accuracy',
                     'Val top-5 accuracy (best epochs only)',
                     color='#9467bd', lw=1.2, linestyle='--', alpha=0.8,
                     marker='o', ms=3),
    ]
    if any(drawn):
        _mark_best(ax, history, 'val_accuracy', 'max')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f"Validation accuracy evolution{title_suffix}",
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right',
                  title='train accuracy is not logged by the trainer',
                  title_fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        written['accuracy'] = save_figure(fig, save_dir, f"{prefix}_accuracy",
                                          plot_format, dpi, show)
    else:
        plt.close(fig)
        print(f"  {WARN} no accuracy metrics in the log - accuracy plot skipped")

    # -- F1 -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    drawn = [
        _plot_series(ax, x, history, 'val_f1_macro', 'F1 macro', color='#1f77b4', lw=1.8),
        _plot_series(ax, x, history, 'val_f1_weighted', 'F1 weighted (best epochs only)',
                     color='#ff7f0e', lw=1.5, marker='o', ms=3),
        _plot_series(ax, x, history, 'val_f1_micro', 'F1 micro (best epochs only)',
                     color='#2ca02c', lw=1.2, linestyle='--', marker='o', ms=3),
        _plot_series(ax, x, history, 'val_cohen_kappa', "Cohen's kappa (best epochs only)",
                     color='#8c564b', lw=1.2, linestyle=':', marker='o', ms=3),
    ]
    if any(drawn):
        _mark_best(ax, history, 'val_f1_macro', 'max')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f"Validation F1 / agreement evolution{title_suffix}",
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        written['f1'] = save_figure(fig, save_dir, f"{prefix}_f1", plot_format, dpi, show)
    else:
        plt.close(fig)

    # -- Learning rate ------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if _plot_series(ax, x, history, 'learning_rate', 'Learning rate',
                    color='#17becf', lw=1.8):
        lr = history.get('learning_rate')
        if np.nanmin(lr) > 0:
            ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning rate', fontsize=12)
        ax.set_title(f"Learning-rate schedule{title_suffix}",
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        fig.tight_layout()
        written['lr'] = save_figure(fig, save_dir, f"{prefix}_lr", plot_format, dpi, show)
    else:
        plt.close(fig)
        print(f"  {WARN} main.py does not log the learning rate - LR plot skipped")

    # -- Confidence ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    drawn = [
        _plot_series(ax, x, history, 'val_mean_confidence', 'Mean confidence',
                     color='#1f77b4', lw=1.8, marker='o', ms=3),
        _plot_series(ax, x, history, 'val_mean_correct_confidence',
                     'Mean confidence (correct)', color='#2ca02c', lw=1.4,
                     marker='o', ms=3),
        _plot_series(ax, x, history, 'val_mean_incorrect_confidence',
                     'Mean confidence (incorrect)', color='#d62728', lw=1.4,
                     marker='o', ms=3),
        _plot_series(ax, x, history, 'val_min_confidence', 'Min confidence',
                     color='#7f7f7f', lw=1.0, linestyle=':', marker='o', ms=3),
    ]
    if any(drawn):
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Confidence', fontsize=12)
        ax.set_title(f"Validation confidence evolution{title_suffix}\n"
                     f"(logged only on best epochs)",
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        written['confidence'] = save_figure(fig, save_dir, f"{prefix}_confidence",
                                            plot_format, dpi, show)
    else:
        plt.close(fig)

    # -- Overview 2x2 -------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    _plot_series(ax, x, history, 'train_loss', 'Train', color='#1f77b4', lw=1.6)
    _plot_series(ax, x, history, 'val_loss', 'Validation', color='#d62728', lw=1.6)
    ax.set_title('Loss', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    _plot_series(ax, x, history, 'val_accuracy', 'Accuracy', color='#2ca02c', lw=1.6)
    _plot_series(ax, x, history, 'val_balanced_accuracy', 'Balanced',
                 color='#ff7f0e', lw=1.3, marker='o', ms=3)
    ax.set_title('Validation accuracy', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    _plot_series(ax, x, history, 'val_f1_macro', 'F1 macro', color='#1f77b4', lw=1.6)
    _plot_series(ax, x, history, 'val_f1_weighted', 'F1 weighted',
                 color='#ff7f0e', lw=1.3, marker='o', ms=3)
    ax.set_title('Validation F1', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    drawn_lr = _plot_series(ax, x, history, 'learning_rate', 'LR',
                            color='#17becf', lw=1.6)
    if drawn_lr:
        lr = history.get('learning_rate')
        if lr is not None and np.nanmin(lr) > 0:
            ax.set_yscale('log')
        ax.set_title('Learning rate', fontweight='bold')
        ax.set_ylabel('LR')
    else:
        # No LR in this log: show the derived epoch wall time instead of an
        # empty panel.
        if _plot_series(ax, x, history, 'epoch_time_s', 'Epoch time',
                        color='#8c564b', lw=1.6):
            ax.set_title('Epoch wall time (from timestamps)', fontweight='bold')
            ax.set_ylabel('Seconds')
            ax.legend(fontsize=9)
        else:
            ax.set_title('Learning rate (not logged)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3, which='both')

    n_ep = history.summary['n_epochs']
    fig.suptitle(
        f"Training overview{title_suffix}  |  {n_ep} epochs "
        f"({history.summary['first_epoch']}-{history.summary['last_epoch']})",
        fontsize=14, fontweight='bold'
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    written['overview'] = save_figure(fig, save_dir, f"{prefix}_overview",
                                      plot_format, dpi, show)

    print(f"{'='*66}")
    return written


# ======================================================================
# Base evaluator: metrics, plots and exports shared by every model
# ======================================================================

class BaseEvaluator:
    """
    Everything that does not depend on *how* the probabilities were produced:
    the extended metric set, the console report, every plot, the CSV/JSON/TXT
    exports and the side-by-side comparison table.

    Subclasses must provide:
        `_predict_split(split)` -> (probabilities, targets, indices, performance)
        `_get_image(split, index)` -> a CHW tensor (only needed for
                                      `plot_misclassified`)

    They must also set `self.num_classes`, `self.split_names` and either
    `self._class_names_override` or a `_class_names(split)` override.
    """

    # -- construction -------------------------------------------------

    def __init__(
        self,
        num_classes: int,
        split_names: Sequence[str],
        class_names: Optional[Sequence[str]] = None,
        model_path: Optional[str] = None,
        split: str = 'val',
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        # The metric and plot layer is pure numpy + sklearn; only the JITrans
        # subclass actually needs torch.
        require_eval_deps('scikit-learn')
        self.num_classes = num_classes
        self.split_names = list(split_names)
        self._class_names_override = list(class_names) if class_names else None
        self.model_path = model_path
        self.split = split
        self.mean = mean or [0.485, 0.456, 0.406]  # ImageNet default
        self.std = std or [0.229, 0.224, 0.225]    # ImageNet default

    # -- hooks --------------------------------------------------------

    def _predict_split(
        self, split_name: str, verbose: bool = True, track_performance: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict]]:
        raise NotImplementedError

    def _get_image(self, split_name: str, index: int) -> torch.Tensor:
        raise NotImplementedError("this evaluator cannot fetch source images")

    def _split_note(self, split_name: str) -> Optional[str]:
        """Optional caveat printed and exported alongside a split's metrics."""
        return None

    # -- utilities ----------------------------------------------------

    def _class_names(self, split: Optional[str] = None) -> List[str]:
        if self._class_names_override:
            return list(self._class_names_override)
        return [f"class_{i}" for i in range(self.num_classes)]

    def tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a normalized CHW tensor into an HWC numpy image in [0, 1]."""
        img = tensor.cpu().detach().numpy()

        if img.ndim == 4:
            img = img.squeeze(0)

        if img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        if img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)

        should_denormalize = (
            self.mean is not None and
            self.std is not None and
            not (np.array(self.mean) == 0).all() and
            not (np.array(self.std) == 255).all()
        )

        if should_denormalize:
            mean = np.array(self.mean)
            std = np.array(self.std)

            if img.ndim == 3 and img.shape[-1] == 3:
                img = img * std + mean
            elif img.ndim == 2:
                img = img * std[0] + mean[0]
            else:
                img = img * std + mean

        return np.clip(img, 0, 1)

    # ------------------------------------------------------------------
    # Extended metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_calibration(
        confidences: np.ndarray,
        correct: np.ndarray,
        n_bins: int = 15
    ) -> Dict[str, Any]:
        """
        Confidence-calibration statistics.

        Expected Calibration Error (ECE) is the sample-weighted mean gap between
        bin accuracy and bin confidence; Maximum Calibration Error (MCE) is the
        largest such gap over non-empty bins.  A perfectly calibrated model has
        ECE = 0 (accuracy inside a bin equals the confidence in that bin).
        """
        confidences = np.asarray(confidences, dtype=float)
        correct = np.asarray(correct, dtype=bool)
        n_samples = len(confidences)

        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_acc = np.full(n_bins, np.nan)
        bin_conf = np.full(n_bins, np.nan)
        bin_count = np.zeros(n_bins, dtype=int)

        ece = 0.0
        mce = 0.0
        signed = 0.0
        for i in range(n_bins):
            lower, upper = edges[i], edges[i + 1]
            # Include the right edge in the last bin so conf == 1.0 is counted
            mask = ((confidences > lower) & (confidences <= upper)) if i > 0 else \
                   ((confidences >= lower) & (confidences <= upper))
            count = int(mask.sum())
            bin_count[i] = count
            if count == 0:
                continue
            acc = float(correct[mask].mean())
            conf = float(confidences[mask].mean())
            bin_acc[i] = acc
            bin_conf[i] = conf
            gap = abs(acc - conf)
            ece += (count / n_samples) * gap
            mce = max(mce, gap)
            signed += (count / n_samples) * (conf - acc)

        return {
            'ece': float(ece),
            'mce': float(mce),
            # Signed counterpart of ECE: > 0 overconfident, < 0 underconfident
            'signed_gap': float(signed),
            'n_bins': n_bins,
            'bin_edges': edges,
            'bin_accuracy': bin_acc,
            'bin_confidence': bin_conf,
            'bin_count': bin_count,
        }

    def _extended_metrics(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        confidences: np.ndarray,
        class_names: List[str],
    ) -> Dict[str, Any]:
        """
        Compute the full metric set for one split.

        Includes macro/micro/weighted averages, Cohen's kappa, Matthews
        correlation coefficient, top-k accuracy, per-class precision / recall /
        F1 / support / specificity / accuracy / mean confidence, and calibration.

        Note: `loss.py` reports `matthews_corrcoef` as 0.0 for multi-class
        problems (it only computes it for the binary case), which is why the
        training log shows 0.0000; here it is computed properly via sklearn.
        """
        labels = list(range(self.num_classes))
        cm = confusion_matrix(targets, predictions, labels=labels)
        correct_mask = (predictions == targets)

        precision_pc = precision_score(targets, predictions, labels=labels,
                                       average=None, zero_division=0)
        recall_pc = recall_score(targets, predictions, labels=labels,
                                 average=None, zero_division=0)
        f1_pc = f1_score(targets, predictions, labels=labels,
                         average=None, zero_division=0)

        total = int(cm.sum())
        per_class: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(class_names[:self.num_classes]):
            tp = int(cm[i, i])
            support = int(cm[i, :].sum())
            predicted = int(cm[:, i].sum())
            fp = predicted - tp
            fn = support - tp
            tn = total - tp - fp - fn
            class_mask = (targets == i)
            per_class[name] = {
                'precision': float(precision_pc[i]),
                'recall': float(recall_pc[i]),
                'f1': float(f1_pc[i]),
                'support': support,
                'predicted_count': predicted,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                'accuracy': float(tp / support) if support > 0 else 0.0,
                'mean_confidence': float(confidences[class_mask].mean()) if class_mask.any() else 0.0,
            }

        metrics: Dict[str, Any] = {
            'accuracy': float(accuracy_score(targets, predictions)),
            'error_rate': float(1.0 - accuracy_score(targets, predictions)),
            'balanced_accuracy': float(balanced_accuracy_score(targets, predictions)),
            'cohen_kappa': float(cohen_kappa_score(targets, predictions)),
            'matthews_corrcoef': float(matthews_corrcoef(targets, predictions)),
            'per_class': per_class,
            'confusion_matrix': cm.tolist(),
            '_confusion_matrix': cm,
        }

        for average in ('macro', 'micro', 'weighted'):
            metrics[f'precision_{average}'] = float(
                precision_score(targets, predictions, labels=labels,
                                average=average, zero_division=0))
            metrics[f'recall_{average}'] = float(
                recall_score(targets, predictions, labels=labels,
                             average=average, zero_division=0))
            metrics[f'f1_{average}'] = float(
                f1_score(targets, predictions, labels=labels,
                         average=average, zero_division=0))

        # Primary precision/recall/f1 kept at the top level: macro for
        # multi-class, binary for 2 classes.
        if self.num_classes > 2:
            metrics['precision'] = metrics['precision_macro']
            metrics['recall'] = metrics['recall_macro']
            metrics['f1_score'] = metrics['f1_macro']
        else:
            metrics['precision'] = float(precision_score(targets, predictions,
                                                         average='binary', zero_division=0))
            metrics['recall'] = float(recall_score(targets, predictions,
                                                   average='binary', zero_division=0))
            metrics['f1_score'] = float(f1_score(targets, predictions,
                                                  average='binary', zero_division=0))

        # Top-k accuracy needs the full probability matrix
        if probabilities is not None and len(probabilities):
            for k in range(1, min(5, self.num_classes) + 1):
                try:
                    metrics[f'top_{k}_accuracy'] = float(
                        top_k_accuracy_score(targets, probabilities, k=k, labels=labels))
                except ValueError as exc:
                    print(f"{WARN} top-{k} accuracy unavailable: {exc}")

            # ROC-AUC / average precision live here (not only in plot_curves)
            # so they always reach the JSON, the per-class CSV and the split
            # comparison; '_curves' is reused by plot_curves().
            curves = compute_curve_data(targets, probabilities, self.num_classes)
            if curves is not None:
                names = class_names[:self.num_classes]
                if curves['skipped']:
                    print(f"{WARN} ROC/PR undefined (no positive or no negative sample) for: "
                          + ', '.join(names[i] for i in curves['skipped']))
                auc_scores = auc_scores_from_curves(curves, names)
                for name in names:
                    per_class[name]['roc_auc'] = auc_scores[f'roc_auc_{name}']
                    per_class[name]['ap'] = auc_scores[f'pr_auc_{name}']
                metrics['auc_scores'] = auc_scores
                metrics['_curves'] = curves

        # Confidence + calibration
        metrics['mean_confidence'] = float(np.mean(confidences))
        metrics['median_confidence'] = float(np.median(confidences))
        metrics['min_confidence'] = float(np.min(confidences))
        metrics['max_confidence'] = float(np.max(confidences))
        metrics['mean_correct_confidence'] = (
            float(confidences[correct_mask].mean()) if correct_mask.any() else 0.0)
        metrics['mean_incorrect_confidence'] = (
            float(confidences[~correct_mask].mean()) if (~correct_mask).any() else 0.0)

        calibration = self._compute_calibration(confidences, correct_mask)
        metrics['calibration'] = {
            'ece': calibration['ece'],
            'mce': calibration['mce'],
            'signed_gap': calibration['signed_gap'],
            'n_bins': calibration['n_bins'],
        }
        metrics['_calibration'] = calibration

        return metrics

    # ------------------------------------------------------------------
    # Evaluation driver
    # ------------------------------------------------------------------

    def evaluate_dataset(
        self,
        verbose: bool = True,
        save_predictions: bool = False,
        output_file: Optional[str] = None,
        track_performance: bool = True,
        split: Optional[str] = None,
    ) -> Dict:
        """Evaluate one loaded split.  See `evaluate_splits` for all of them."""
        split_name = split or self.split_names[0]
        if split_name not in self.split_names:
            raise ValueError(
                f"Split '{split_name}' is not loaded. Available: {self.split_names}"
            )

        results = self._evaluate_one(
            split_name,
            verbose=verbose,
            save_predictions=save_predictions,
            track_performance=track_performance,
        )

        if output_file:
            self._write_json(results, output_file)

        return results

    def evaluate_splits(
        self,
        verbose: bool = True,
        save_predictions: bool = False,
        track_performance: bool = True,
    ) -> Dict[str, Dict]:
        """
        Evaluate every loaded split independently, reusing the single loaded
        model (the checkpoint is never read twice).

        Returns:
            Dict split-name -> result dict.
        """
        results_by_split: Dict[str, Dict] = {}
        for name in self.split_names:
            if verbose:
                print(f"\n{'#'*66}")
                print(f"#  EVALUATING SPLIT: {name.upper()}")
                print(f"{'#'*66}")
            results_by_split[name] = self._evaluate_one(
                name,
                verbose=verbose,
                save_predictions=save_predictions,
                track_performance=track_performance,
            )

        if len(results_by_split) > 1 and verbose:
            self.print_split_comparison(results_by_split)

        return results_by_split

    def _evaluate_one(
        self,
        split_name: str,
        verbose: bool = True,
        save_predictions: bool = False,
        track_performance: bool = True,
    ) -> Dict:
        """Get the probabilities for one split and compute every metric."""
        probabilities, targets, indices, performance = self._predict_split(
            split_name, verbose=verbose, track_performance=track_performance
        )

        probabilities = np.asarray(probabilities)
        targets = np.asarray(targets)
        predictions = probabilities.argmax(axis=1)
        confidences = probabilities.max(axis=1)

        class_names = self._class_names(split_name)

        metrics = self._extended_metrics(
            targets, predictions, probabilities, confidences, class_names
        )

        results: Dict[str, Any] = {
            'split': split_name,
            'split_mode': self.split,
            'model': getattr(self, 'MODEL_NAME', 'model'),
            'model_path': self.model_path,
            'n_samples': len(targets),
            'num_classes': self.num_classes,
            'class_names': class_names,
            'performance': performance,
            # Arrays kept in memory for the plots; '_'-prefixed keys are
            # stripped before JSON serialisation.
            '_probabilities': probabilities,
            '_targets': targets,
            '_predictions': predictions,
            '_confidences': confidences,
            '_indices': np.asarray(indices),
            'predictions': predictions.tolist() if save_predictions else None,
            'targets': targets.tolist() if save_predictions else None,
            'confidences': confidences.tolist() if save_predictions else None,
        }
        results.update(metrics)

        note = self._split_note(split_name)
        if note:
            results['note'] = note

        if verbose:
            self._print_evaluation_results(results)

        return results

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------

    def _print_evaluation_results(self, results: Dict):
        """Print the full metric report for one split."""
        split_name = results['split']
        performance = results.get('performance')
        class_names = results['class_names']

        print(f"\n{'='*66}")
        print(f"DATASET EVALUATION RESULTS  [{results.get('model', 'model')} | "
              f"split={split_name.upper()}]")
        print(f"{'='*66}")
        print(f"Total Images: {results['n_samples']}")

        if results.get('note'):
            print(f"{WARN} {results['note']}")

        if performance:
            print(f"Processing Time: {performance['total_time_s']:.2f}s "
                  f"({performance['mean_time_per_image_ms']:.3f}ms per image)")
            print(f"Throughput: {performance['throughput_imgs_per_sec']:.2f} images/sec")

        print(f"\n--- Overall ---")
        print(f"Accuracy:          {results['accuracy']:.4f}")
        print(f"Error rate:        {results['error_rate']:.4f}")
        print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
        print(f"Cohen's kappa:     {results['cohen_kappa']:.4f}")
        print(f"Matthews corrcoef: {results['matthews_corrcoef']:.4f}")

        print(f"\n--- Averaged (macro / micro / weighted) ---")
        print(f"{'':<12}{'macro':>10}{'micro':>10}{'weighted':>10}")
        for metric in ('precision', 'recall', 'f1'):
            row = ''.join(f"{results[f'{metric}_{avg}']:>10.4f}"
                          for avg in ('macro', 'micro', 'weighted'))
            print(f"{metric:<12}{row}")

        topk = sorted(k for k in results if re.fullmatch(r'top_\d+_accuracy', k))
        if topk:
            print(f"\n--- Top-k accuracy ---")
            for key in topk:
                print(f"{key:<18} {results[key]:.4f}")

        auc_scores = results.get('auc_scores')
        if auc_scores:
            print(f"\n--- ROC-AUC / AP (macro / micro / weighted) ---")
            print(f"{'':<12}{'macro':>10}{'micro':>10}{'weighted':>10}")
            for kind, label in (('roc', 'ROC-AUC'), ('pr', 'AP')):
                row = ''.join(f"{auc_scores[f'{kind}_auc_{avg}']:>10.4f}"
                              for avg in ('macro', 'micro', 'weighted'))
                print(f"{label:<12}{row}")

        print(f"\n--- Confidence ---")
        print(f"Mean:               {results['mean_confidence']:.4f}")
        print(f"Median:             {results['median_confidence']:.4f}")
        print(f"Min / Max:          {results['min_confidence']:.4f} / {results['max_confidence']:.4f}")
        print(f"Mean (correct):     {results['mean_correct_confidence']:.4f}")
        print(f"Mean (incorrect):   {results['mean_incorrect_confidence']:.4f}")
        calibration = results['calibration']
        print(f"ECE ({calibration['n_bins']} bins):       {calibration['ece']:.4f}")
        print(f"MCE:                {calibration['mce']:.4f}")
        tendency = 'overconfident' if calibration['signed_gap'] > 0 else 'underconfident'
        print(f"Signed gap:         {calibration['signed_gap']:+.4f}  ({tendency})")

        print(f"\n--- Per class ---")
        header = (f"  {'class':<16}{'prec':>8}{'recall':>8}{'f1':>8}"
                  f"{'spec':>8}{'support':>9}{'pred':>8}{'conf':>8}")
        print(header)
        print(f"  {'-'*(len(header)-2)}")
        for name in class_names:
            stats = results['per_class'].get(name)
            if stats is None:
                continue
            print(f"  {name:<16}{stats['precision']:>8.4f}{stats['recall']:>8.4f}"
                  f"{stats['f1']:>8.4f}{stats['specificity']:>8.4f}"
                  f"{stats['support']:>9}{stats['predicted_count']:>8}"
                  f"{stats['mean_confidence']:>8.4f}")

        print("\nDetailed Classification Report:")
        print(classification_report(
            results['_targets'],
            results['_predictions'],
            labels=list(range(self.num_classes)),
            target_names=class_names,
            zero_division=0
        ))

        print("Confusion matrix (rows = true, cols = predicted):")
        cm = results['_confusion_matrix']
        width = max(8, max(len(n) for n in class_names) + 1)
        print(' ' * width + ''.join(f"{n[:7]:>8}" for n in class_names))
        for i, name in enumerate(class_names):
            print(f"{name:<{width}}" + ''.join(f"{int(v):>8}" for v in cm[i]))

        if performance:
            print(f"\n{'='*66}")
            print("PERFORMANCE METRICS")
            print(f"{'='*66}")
            print(f"Device:      {performance['device']}")
            if performance.get('batch_size') is not None:
                print(f"Batch Size:  {performance['batch_size']}")
                print(f"Num Batches: {performance['num_batches']}")
            print(f"\nTiming:")
            if performance.get('mean_batch_time_ms') is not None:
                print(f"  Mean Batch Time:     {performance['mean_batch_time_ms']:.2f} +/- "
                      f"{performance['std_batch_time_ms']:.2f} ms")
                print(f"  Median Batch Time:   {performance['median_batch_time_ms']:.2f} ms")
            print(f"  Mean Time per Image: {performance['mean_time_per_image_ms']:.4f} ms")
            print(f"  Total Throughput:    {performance['throughput_imgs_per_sec']:.2f} imgs/s")
            if performance.get('baseline_allocated_mb') is not None:
                print(f"\nMemory:")
                print(f"  Baseline allocated:  {performance['baseline_allocated_mb']:.2f} MB")
                if performance.get('baseline_reserved_mb') is not None:
                    print(f"  Baseline reserved:   {performance['baseline_reserved_mb']:.2f} MB  "
                          f"(total CUDA pool)")
                print(f"  Peak allocated:      {performance['peak_allocated_mb']:.2f} MB")
                if performance.get('peak_reserved_mb') is not None:
                    print(f"  Peak reserved:       {performance['peak_reserved_mb']:.2f} MB")
                print(f"  Avg allocated:       {performance['avg_allocated_mb']:.2f} MB")
                print(f"  Inference overhead:  {performance['inference_overhead_mb']:.2f} MB  "
                      f"(peak - baseline)")
                print(f"  Per image:           {performance['memory_per_image_mb']:.4f} MB")

        print(f"{'='*66}")

    # ------------------------------------------------------------------
    # Split / model comparison
    # ------------------------------------------------------------------

    COMPARISON_METRICS = [
        ('n_samples', 'Samples', '{:,}'),
        ('accuracy', 'Accuracy', '{:.4f}'),
        ('error_rate', 'Error rate', '{:.4f}'),
        ('balanced_accuracy', 'Balanced accuracy', '{:.4f}'),
        ('precision_macro', 'Precision (macro)', '{:.4f}'),
        ('recall_macro', 'Recall (macro)', '{:.4f}'),
        ('f1_macro', 'F1 (macro)', '{:.4f}'),
        ('f1_weighted', 'F1 (weighted)', '{:.4f}'),
        ('cohen_kappa', "Cohen's kappa", '{:.4f}'),
        ('matthews_corrcoef', 'Matthews corrcoef', '{:.4f}'),
        ('top_5_accuracy', 'Top-5 accuracy', '{:.4f}'),
        ('mean_confidence', 'Mean confidence', '{:.4f}'),
        ('mean_correct_confidence', 'Confidence (correct)', '{:.4f}'),
        ('mean_incorrect_confidence', 'Confidence (incorrect)', '{:.4f}'),
    ]

    def print_split_comparison(
        self,
        results_by_key: Dict[str, Dict],
        out_dir: Optional[str] = None,
        title: str = 'SPLIT COMPARISON',
        csv_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Print a side-by-side table, and optionally write it to CSV.

        The keys are free-form: `evaluate_splits` passes split names, while
        `--compare` passes model names (jit / mamba / stacking).

        Returns:
            Path of the written CSV, or None.
        """
        keys = list(results_by_key.keys())
        rows: List[Tuple[str, List[str]]] = []

        for key, label, fmt in self.COMPARISON_METRICS:
            values = []
            present = False
            for column in keys:
                value = results_by_key[column].get(key)
                if value is None:
                    values.append('-')
                else:
                    present = True
                    values.append(fmt.format(value))
            if present:
                rows.append((label, values))

        # AUC rows (present whenever probabilities were available)
        for key, label in (('roc_auc_macro', 'ROC-AUC (macro)'),
                           ('roc_auc_micro', 'ROC-AUC (micro)'),
                           ('roc_auc_weighted', 'ROC-AUC (weighted)'),
                           ('pr_auc_macro', 'PR-AUC (macro)'),
                           ('pr_auc_micro', 'PR-AUC (micro)'),
                           ('pr_auc_weighted', 'PR-AUC (weighted)')):
            values, present = [], False
            for column in keys:
                auc_scores = results_by_key[column].get('auc_scores') or {}
                value = auc_scores.get(key)
                if value is None:
                    values.append('-')
                else:
                    present = True
                    values.append(f"{value:.4f}")
            if present:
                rows.append((label, values))

        # Calibration rows
        for key, label, fmt in (('ece', 'ECE', '{:.4f}'),
                                ('mce', 'MCE', '{:.4f}'),
                                ('signed_gap', 'Signed gap', '{:+.4f}')):
            values, present = [], False
            for column in keys:
                calibration = results_by_key[column].get('calibration') or {}
                value = calibration.get(key)
                if value is None:
                    values.append('-')
                else:
                    present = True
                    values.append(fmt.format(value))
            if present:
                rows.append((label, values))

        if not rows:
            return None

        # --compare results carry a readable name; split results fall back to the key
        headers = [results_by_key[k].get('display_name') or k.upper() for k in keys]
        label_width = max(len(label) for label, _ in rows) + 2
        col_width = max(14, max(len(h) for h in headers) + 2)

        print(f"\n{'='*66}")
        print(title)
        print(f"{'='*66}")
        print(f"{'Metric':<{label_width}}" + ''.join(f"{h:>{col_width}}" for h in headers))
        print('-' * (label_width + col_width * len(keys)))
        for label, values in rows:
            print(f"{label:<{label_width}}" + ''.join(f"{v:>{col_width}}" for v in values))
        for column in keys:
            note = results_by_key[column].get('note')
            if note:
                print(f"{WARN} {results_by_key[column].get('display_name') or column}: {note}")
        print(f"{'='*66}")

        if not out_dir:
            return None

        os.makedirs(out_dir, exist_ok=True)
        name = csv_name or f"comparison_{'_vs_'.join(keys)}.csv"
        path = os.path.join(out_dir, name)
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['metric'] + [results_by_key[k].get('display_name') or k
                                          for k in keys])
            for label, values in rows:
                writer.writerow([label] + values)
        print(f"  {OK} saved {ARROW} {path}")
        return path

    # ------------------------------------------------------------------
    # Curve plots - ROC and Precision-Recall
    # ------------------------------------------------------------------

    def plot_curves(
        self,
        results: Dict,
        save_dir: Optional[str] = None,
        show: bool = False,
        dpi: int = 150,
        plot_format: str = 'pdf',
    ) -> Dict[str, float]:
        """
        Plot and optionally save the multiclass AUC-ROC and Precision-Recall
        curves: one-vs-rest curve of every class plus the micro- and
        macro-average curves.  The PR figure also shows iso-F1 curves.
        When the model is good enough for the curves to hug a corner
        (macro AUC >= 0.95, macro AP >= 0.9) the corner is magnified in an inset.
        With two classes only the positive-class curve is drawn.

        Args:
            results:     Dict returned by evaluate_dataset().  Must contain the
                         '_probabilities' and '_targets' numpy arrays that are
                         stored there automatically.
            save_dir:    Directory where the two files are saved.
                         If None the figures are not saved to disk.
            show:        Call plt.show() after building each figure.
            dpi:         Resolution of saved raster figures (PNG only).
            plot_format: 'pdf' (default), 'png' or 'both'.

        Returns:
            Dict with per-class and macro / micro / weighted AUC scores, e.g.:
            {
              'roc_auc_dry': 0.999,
              'roc_auc_macro': 0.998,
              'roc_auc_micro': 0.999,
              'pr_auc_dry': 0.997,
              'pr_auc_macro': 0.996,
              ...
            }
            The same dict is also stored in `results['auc_scores']`.
        """
        probs   = results.get('_probabilities')
        targets = results.get('_targets')

        if probs is None or len(probs) == 0:
            raise ValueError(
                "results dict does not contain '_probabilities'. "
                "Run evaluate_dataset() first - it stores probabilities automatically."
            )

        split       = results.get('split', 'split')
        prefix      = results.get('model', 'model')
        class_names = results.get('class_names') or self._class_names(split)

        # Curves are normally computed once in _extended_metrics()
        curves = results.get('_curves')
        if curves is None:
            curves = compute_curve_data(targets, probs, self.num_classes)
            if curves is None:
                raise ValueError("no class has both positive and negative samples")
            if curves['skipped']:
                print(f"{WARN} ROC/PR undefined (no positive or no negative sample) for: "
                      + ', '.join(class_names[i] for i in curves['skipped']))
        auc_scores = auc_scores_from_curves(curves, class_names[:self.num_classes])

        binary  = curves['macro'] is None
        classes = [i for i in curves['valid'] if not binary or i == 1] or curves['valid'][:1]

        # == Colour palette ==============================================
        cmap   = plt.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(self.num_classes)]

        # == Figure 1: AUC-ROC ===========================================
        fig_roc, ax_roc = plt.subplots(figsize=(8, 7))
        roc_lines = []

        for i in classes:
            fpr, tpr, roc_auc = curves['roc'][i]
            style = dict(color=colors[i], lw=1.8)
            ax_roc.plot(fpr, tpr, label=f"{class_names[i]}  (AUC = {roc_auc:.4f})", **style)
            roc_lines.append((fpr, tpr, style))

        if binary:
            title = f"AUC = {auc_scores['roc_auc_macro']:.4f}"
        else:
            for avg, style in (('micro', MICRO_STYLE), ('macro', MACRO_STYLE)):
                fpr, tpr, roc_auc = curves[avg]['roc']
                ax_roc.plot(fpr, tpr, label=f"{avg.capitalize()} avg  (AUC = {roc_auc:.4f})",
                            **style)
                roc_lines.append((fpr, tpr, style))
            title = (f"Micro AUC = {auc_scores['roc_auc_micro']:.4f}  |  "
                     f"Macro AUC = {auc_scores['roc_auc_macro']:.4f}")

        ax_roc.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.5, label="Chance level  (AUC = 0.5)")

        if auc_scores['roc_auc_macro'] >= 0.95:
            # Top-left corner; the lower bound follows the worst curve
            x_max = 0.05
            y_min = min(_value_at(0.1 * x_max, x, y) for x, y, _ in roc_lines)
            _add_zoom_inset(ax_roc, roc_lines, (0, x_max), (y_min, 1.0),
                            [0.35, 0.35, 0.4, 0.4])

        ax_roc.set_xlim([-0.01, 1.01])
        ax_roc.set_ylim([-0.01, 1.02])
        ax_roc.set_xlabel("False Positive Rate", fontsize=12)
        ax_roc.set_ylabel("True Positive Rate", fontsize=12)
        ax_roc.set_title(f"ROC Curves - {prefix} / {split.upper()}\n{title}",
                         fontsize=13, fontweight='bold')
        ax_roc.legend(loc="lower right", fontsize=9)
        ax_roc.grid(True, alpha=0.3)
        fig_roc.tight_layout()
        save_figure(fig_roc, save_dir, f"roc_curves_{prefix}_{split}",
                    plot_format, dpi, show)

        # == Figure 2: AUC-Precision-Recall ==============================
        fig_pr, ax_pr = plt.subplots(figsize=(8, 7))
        _draw_iso_f1(ax_pr)
        pr_lines = []

        for i in classes:
            rec, prec, ap = curves['pr'][i]
            style = dict(color=colors[i], lw=1.8, drawstyle='steps-post')
            ax_pr.plot(rec, prec, label=f"{class_names[i]}  (AP = {ap:.4f})", **style)
            pr_lines.append((rec, prec, style))

        if binary:
            title = f"AP = {auc_scores['pr_auc_macro']:.4f}"
        else:
            rec, prec, ap = curves['micro']['pr']
            style = dict(MICRO_STYLE, drawstyle='steps-post')
            ax_pr.plot(rec, prec, label=f"Micro avg  (AP = {ap:.4f})", **style)
            pr_lines.append((rec, prec, style))
            rec, prec, ap = curves['macro']['pr']
            ax_pr.plot(rec, prec, label=f"Macro avg  (AP = {ap:.4f})", **MACRO_STYLE)
            pr_lines.append((rec, prec, MACRO_STYLE))
            title = (f"Micro AP = {auc_scores['pr_auc_micro']:.4f}  |  "
                     f"Macro AP = {auc_scores['pr_auc_macro']:.4f}")

        if auc_scores['pr_auc_macro'] >= 0.9:
            # Top-right corner
            y_min = min(_value_at(0.9, x, y) for x, y, _ in pr_lines)
            _add_zoom_inset(ax_pr, pr_lines, (0.8, 1.0), (y_min, 1.0),
                            [0.3, 0.35, 0.4, 0.4])

        ax_pr.set_xlim([-0.01, 1.01])
        ax_pr.set_ylim([-0.01, 1.02])
        ax_pr.set_xlabel("Recall", fontsize=12)
        ax_pr.set_ylabel("Precision", fontsize=12)
        ax_pr.set_title(f"Precision-Recall Curves - {prefix} / {split.upper()}\n{title}",
                        fontsize=13, fontweight='bold')
        ax_pr.legend(loc="lower left", fontsize=9)
        ax_pr.grid(True, alpha=0.3)
        fig_pr.tight_layout()
        save_figure(fig_pr, save_dir, f"pr_curves_{prefix}_{split}",
                    plot_format, dpi, show)

        # == Summary print ================================================
        print(f"\n{'='*60}")
        print(f"AUC SCORES  [{prefix} | split={split.upper()}]")
        print(f"{'='*60}")
        col = 22
        print(f"  {'Class':<{col}} {'ROC-AUC':>10}   {'PR-AUC (AP)':>12}")
        print(f"  {'-'*col} {'-'*10}   {'-'*12}")
        for cls_name in class_names:
            r = auc_scores.get(f'roc_auc_{cls_name}', float('nan'))
            p = auc_scores.get(f'pr_auc_{cls_name}',  float('nan'))
            print(f"  {cls_name:<{col}} {r:>10.4f}   {p:>12.4f}")
        print(f"  {'='*col} {'='*10}   {'='*12}")
        for avg in ('macro', 'micro', 'weighted'):
            label = f"{avg.capitalize()} average"
            print(f"  {label:<{col}} {auc_scores[f'roc_auc_{avg}']:>10.4f}   "
                  f"{auc_scores[f'pr_auc_{avg}']:>12.4f}")
        print(f"{'='*60}")

        results['auc_scores'] = auc_scores
        return auc_scores

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def _confusion_matrix_figure(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str,
        normalized: bool,
    ):
        """Build one confusion-matrix heatmap figure."""
        n = len(class_names)
        data = cm.astype(float)
        if normalized:
            row_sums = data.sum(axis=1, keepdims=True)
            with np.errstate(invalid='ignore', divide='ignore'):
                data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums > 0)

        size = max(6.5, 1.15 * n + 2.5)
        fig, ax = plt.subplots(figsize=(size, size * 0.88))
        image = ax.imshow(data, cmap='Blues', vmin=0,
                          vmax=1.0 if normalized else data.max())

        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label('Fraction of true class' if normalized else 'Samples',
                           fontsize=10)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(class_names, fontsize=10)
        ax.set_xlabel('Predicted label', fontsize=12)
        ax.set_ylabel('True label', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')

        # Annotate cells; flip text colour on dark cells for readability
        threshold = (1.0 if normalized else data.max()) * 0.55
        fontsize = 10 if n <= 8 else 7
        for i in range(n):
            for j in range(n):
                if normalized:
                    text = f"{data[i, j]*100:.1f}%"
                else:
                    text = f"{int(cm[i, j]):,}"
                ax.text(j, i, text, ha='center', va='center', fontsize=fontsize,
                        color='white' if data[i, j] > threshold else 'black')

        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.2)
        ax.tick_params(which='minor', length=0)
        fig.tight_layout()
        return fig

    def plot_confusion_matrix(
        self,
        results: Dict,
        save_dir: Optional[str] = None,
        show: bool = False,
        dpi: int = 150,
        plot_format: str = 'pdf',
    ) -> Dict[str, List[str]]:
        """
        Plot the confusion matrix twice: raw counts and row-normalized
        (row-normalized diagonal = per-class recall).
        """
        cm = results.get('_confusion_matrix')
        if cm is None:
            cm = np.array(results['confusion_matrix'])
        split = results.get('split', 'split')
        prefix = results.get('model', 'model')
        class_names = results.get('class_names') or self._class_names(split)
        accuracy = results.get('accuracy')

        print(f"\nConfusion matrix plots [{prefix} | split={split.upper()}]")

        suffix = f"  |  accuracy = {accuracy:.4f}" if accuracy is not None else ""
        fig = self._confusion_matrix_figure(
            cm, class_names,
            f"Confusion Matrix - {prefix} / {split.upper()}{suffix}",
            normalized=False,
        )
        written = {'counts': save_figure(
            fig, save_dir, f"confusion_matrix_{prefix}_{split}",
            plot_format, dpi, show)}

        balanced = results.get('balanced_accuracy')
        suffix = f"  |  balanced accuracy = {balanced:.4f}" if balanced is not None else ""
        fig = self._confusion_matrix_figure(
            cm, class_names,
            f"Confusion Matrix (row-normalized) - {prefix} / {split.upper()}{suffix}",
            normalized=True,
        )
        written['normalized'] = save_figure(
            fig, save_dir, f"confusion_matrix_normalized_{prefix}_{split}",
            plot_format, dpi, show)

        return written

    # ------------------------------------------------------------------
    # Confidence histogram
    # ------------------------------------------------------------------

    def plot_confidence_histogram(
        self,
        results: Dict,
        save_dir: Optional[str] = None,
        show: bool = False,
        dpi: int = 150,
        plot_format: str = 'pdf',
        bins: int = 40,
    ) -> List[str]:
        """
        Overlaid confidence histograms for correct vs incorrect predictions,
        plus the selective-prediction curve.

        A well-behaved model puts correct predictions at high confidence and
        errors at low confidence; overlap in the high-confidence region means
        confidently wrong predictions.
        """
        confidences = results['_confidences']
        correct = results['_predictions'] == results['_targets']
        split = results.get('split', 'split')
        prefix = results.get('model', 'model')

        fig, (ax_hist, ax_cum) = plt.subplots(1, 2, figsize=(13, 5))

        edges = np.linspace(confidences.min(), 1.0, bins + 1)
        ax_hist.hist(confidences[correct], bins=edges, color='#2ca02c', alpha=0.65,
                     label=f"Correct (n={int(correct.sum()):,})")
        ax_hist.hist(confidences[~correct], bins=edges, color='#d62728', alpha=0.65,
                     label=f"Incorrect (n={int((~correct).sum()):,})")
        if correct.any():
            ax_hist.axvline(confidences[correct].mean(), color='#2ca02c',
                            linestyle='--', lw=1.5,
                            label=f"mean correct = {confidences[correct].mean():.4f}")
        if (~correct).any():
            ax_hist.axvline(confidences[~correct].mean(), color='#d62728',
                            linestyle='--', lw=1.5,
                            label=f"mean incorrect = {confidences[~correct].mean():.4f}")
        ax_hist.set_xlabel('Predicted confidence', fontsize=12)
        ax_hist.set_ylabel('Number of samples', fontsize=12)
        ax_hist.set_title('Confidence distribution', fontsize=12, fontweight='bold')
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.3)

        # Accuracy vs confidence threshold - how much can be auto-accepted
        thresholds = np.linspace(confidences.min(), confidences.max(), 100)
        coverage, selective_acc = [], []
        for threshold in thresholds:
            mask = confidences >= threshold
            coverage.append(mask.mean())
            selective_acc.append(correct[mask].mean() if mask.any() else np.nan)
        ax_cum.plot(thresholds, selective_acc, color='#1f77b4', lw=2,
                    label='Accuracy above threshold')
        ax_cum.plot(thresholds, coverage, color='#ff7f0e', lw=2, linestyle='--',
                    label='Coverage (fraction kept)')
        ax_cum.set_xlabel('Confidence threshold', fontsize=12)
        ax_cum.set_ylabel('Accuracy / coverage', fontsize=12)
        ax_cum.set_title('Selective prediction', fontsize=12, fontweight='bold')
        ax_cum.legend(fontsize=9, loc='center left')
        ax_cum.grid(True, alpha=0.3)

        fig.suptitle(f"Confidence analysis - {prefix} / {split.upper()}",
                     fontsize=13, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        print(f"\nConfidence histogram [{prefix} | split={split.upper()}]")
        return save_figure(fig, save_dir, f"confidence_hist_{prefix}_{split}",
                           plot_format, dpi, show)

    # ------------------------------------------------------------------
    # Per-class metrics / class distribution
    # ------------------------------------------------------------------

    def plot_per_class_metrics(
        self,
        results: Dict,
        save_dir: Optional[str] = None,
        show: bool = False,
        dpi: int = 150,
        plot_format: str = 'pdf',
    ) -> List[str]:
        """Grouped bar chart of per-class precision, recall and F1."""
        split = results.get('split', 'split')
        prefix = results.get('model', 'model')
        class_names = results.get('class_names') or self._class_names(split)
        per_class = results['per_class']

        names = [n for n in class_names if n in per_class]
        x = np.arange(len(names))
        width = 0.26

        fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(names) + 3), 6))
        series = [
            ('Precision', [per_class[n]['precision'] for n in names], '#1f77b4'),
            ('Recall',    [per_class[n]['recall']    for n in names], '#ff7f0e'),
            ('F1',        [per_class[n]['f1']        for n in names], '#2ca02c'),
        ]
        for offset, (label, values, color) in zip((-width, 0.0, width), series):
            bars = ax.bar(x + offset, values, width, label=label, color=color)
            ax.bar_label(bars, fmt='%.3f', fontsize=7, rotation=90, padding=2)

        macro_f1 = results.get('f1_macro')
        if macro_f1 is not None:
            ax.axhline(macro_f1, color='black', linestyle=':', lw=1.2,
                       label=f"F1 macro = {macro_f1:.4f}")

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{n}\n(n={per_class[n]['support']:,})" for n in names],
            fontsize=9
        )
        ax.set_ylim(0, 1.12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f"Per-class metrics - {prefix} / {split.upper()}",
                     fontsize=13, fontweight='bold')
        # Below the axes: the bars are tall for every class, so any in-plot
        # legend would cover data.
        ax.legend(fontsize=9, ncol=4, loc='upper center',
                  bbox_to_anchor=(0.5, -0.13), frameon=False)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()

        print(f"\nPer-class metrics plot [{prefix} | split={split.upper()}]")
        return save_figure(fig, save_dir, f"per_class_metrics_{prefix}_{split}",
                           plot_format, dpi, show)

    def plot_class_distribution(
        self,
        results: Dict,
        save_dir: Optional[str] = None,
        show: bool = False,
        dpi: int = 150,
        plot_format: str = 'pdf',
    ) -> List[str]:
        """
        True vs predicted counts per class - makes over/under-prediction and the
        dataset imbalance visible at a glance.
        """
        split = results.get('split', 'split')
        prefix = results.get('model', 'model')
        class_names = results.get('class_names') or self._class_names(split)
        per_class = results['per_class']
        names = [n for n in class_names if n in per_class]

        support = [per_class[n]['support'] for n in names]
        predicted = [per_class[n]['predicted_count'] for n in names]

        x = np.arange(len(names))
        width = 0.38

        fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(names) + 3), 5.5))
        bars_true = ax.bar(x - width / 2, support, width, label='True (support)',
                           color='#4c72b0')
        bars_pred = ax.bar(x + width / 2, predicted, width, label='Predicted',
                           color='#dd8452')
        ax.bar_label(bars_true, fmt='%d', fontsize=8, padding=2)
        ax.bar_label(bars_pred, fmt='%d', fontsize=8, padding=2)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
        ax.set_ylabel('Number of samples', fontsize=12)
        ax.set_title(f"Class distribution: true vs predicted - {prefix} / {split.upper()}",
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()

        print(f"\nClass distribution plot [{prefix} | split={split.upper()}]")
        return save_figure(fig, save_dir, f"class_distribution_{prefix}_{split}",
                           plot_format, dpi, show)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def plot_calibration(
        self,
        results: Dict,
        save_dir: Optional[str] = None,
        show: bool = False,
        dpi: int = 150,
        plot_format: str = 'pdf',
    ) -> List[str]:
        """
        Reliability diagram: accuracy vs confidence per bin, with the gap to
        perfect calibration highlighted, plus ECE / MCE in the title.
        """
        split = results.get('split', 'split')
        prefix = results.get('model', 'model')
        calibration = results.get('_calibration')
        if calibration is None:
            calibration = self._compute_calibration(
                results['_confidences'],
                results['_predictions'] == results['_targets'],
            )

        edges = calibration['bin_edges']
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)
        accuracy = calibration['bin_accuracy']
        confidence = calibration['bin_confidence']
        counts = calibration['bin_count']

        fig, (ax_rel, ax_count) = plt.subplots(
            2, 1, figsize=(8, 8), sharex=True,
            gridspec_kw={'height_ratios': [3, 1]}
        )

        ax_rel.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfect calibration')
        valid = ~np.isnan(accuracy)
        ax_rel.bar(centers[valid], accuracy[valid], width=widths[valid] * 0.92,
                   color='#4c72b0', edgecolor='white', label='Accuracy', zorder=3)
        # The gap can point either way (over- or underconfident), so draw it
        # unfilled: a translucent fill over the solid accuracy bar would be
        # ambiguous exactly in the underconfident case.
        gap = np.where(valid, np.nan_to_num(confidence) - np.nan_to_num(accuracy), np.nan)
        ax_rel.bar(centers[valid], gap[valid], width=widths[valid] * 0.92,
                   bottom=accuracy[valid], fill=False, edgecolor='#d62728',
                   linewidth=0.9, hatch='//', label='Gap (conf - acc)', zorder=5)
        ax_rel.plot(confidence[valid], accuracy[valid], 'o-', color='#2ca02c',
                    lw=1.4, ms=4, label='Bin (conf, acc)', zorder=6)

        ax_rel.set_xlim(0, 1)
        ax_rel.set_ylim(0, 1.02)
        ax_rel.set_ylabel('Accuracy', fontsize=12)
        # Signed gap tells over- from underconfidence, which the unsigned ECE cannot.
        signed_gap = calibration['signed_gap']
        tendency = 'overconfident' if signed_gap > 0 else 'underconfident'
        ax_rel.set_title(
            f"Reliability diagram - {prefix} / {split.upper()}  |  "
            f"ECE = {calibration['ece']:.4f}   MCE = {calibration['mce']:.4f}\n"
            f"mean signed gap = {signed_gap:+.4f}  ({tendency})",
            fontsize=12, fontweight='bold'
        )
        ax_rel.legend(fontsize=9, loc='upper left')
        ax_rel.grid(True, alpha=0.3)

        ax_count.bar(centers, counts, width=widths * 0.92, color='#7f7f7f')
        ax_count.set_yscale('log')
        ax_count.set_xlabel('Predicted confidence', fontsize=12)
        ax_count.set_ylabel('Samples\n(log)', fontsize=10)
        ax_count.grid(True, alpha=0.3)

        fig.tight_layout()
        print(f"\nCalibration plot [{prefix} | split={split.upper()}]  "
              f"ECE={calibration['ece']:.4f}  MCE={calibration['mce']:.4f}")
        return save_figure(fig, save_dir, f"calibration_{prefix}_{split}",
                           plot_format, dpi, show)

    # ------------------------------------------------------------------
    # Misclassified examples
    # ------------------------------------------------------------------

    def plot_misclassified(
        self,
        results: Dict,
        save_dir: Optional[str] = None,
        show: bool = False,
        dpi: int = 150,
        plot_format: str = 'pdf',
        top_n: int = 16,
        order: str = 'confidence',
    ) -> List[str]:
        """
        Grid of the worst misclassifications, i.e. the errors the model was most
        confident about - these are usually the informative ones.

        Args:
            top_n: How many images to show (grid is sized automatically).
            order: 'confidence' -> most confident errors first (default);
                   'random'     -> a random sample of errors.
        """
        split = results.get('split', 'split')
        prefix = results.get('model', 'model')

        predictions = results['_predictions']
        targets = results['_targets']
        confidences = results['_confidences']
        indices = results['_indices']
        class_names = results.get('class_names') or self._class_names(split)

        wrong = np.flatnonzero(predictions != targets)
        if wrong.size == 0:
            print(f"{OK} No misclassified samples on split '{split}' - nothing to plot.")
            return []

        # Fail early and clearly if this evaluator cannot reach the images at
        # all, instead of producing a grid of "image unavailable" cells.
        try:
            self._get_image(split, int(indices[wrong[0]]))
        except NotImplementedError as exc:
            print(f"{WARN} Misclassified plot skipped: {exc}")
            return []
        except Exception as exc:
            print(f"{WARN} Misclassified plot skipped: cannot read source images ({exc})")
            return []

        if order == 'random':
            rng = np.random.default_rng(42)
            selected = rng.choice(wrong, size=min(top_n, wrong.size), replace=False)
        else:
            selected = wrong[np.argsort(-confidences[wrong])][:top_n]

        n = len(selected)
        n_cols = int(np.ceil(np.sqrt(n)))
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(3.0 * n_cols, 3.4 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        for ax, position in zip(axes, selected):
            dataset_index = int(indices[position])
            try:
                image_tensor = self._get_image(split, dataset_index)
                ax.imshow(self.tensor_to_image(image_tensor),
                          cmap=None if image_tensor.shape[0] == 3 else 'gray')
            except Exception as exc:
                ax.text(0.5, 0.5, f"image unavailable\n{exc}", ha='center',
                        va='center', fontsize=7, wrap=True)
            true_name = class_names[int(targets[position])]
            pred_name = class_names[int(predictions[position])]
            ax.set_title(
                f"true: {true_name}\npred: {pred_name} ({confidences[position]:.3f})",
                fontsize=9, color='#b22222'
            )
            ax.axis('off')

        for ax in axes[n:]:
            ax.axis('off')

        error_rate = wrong.size / len(targets)
        fig.suptitle(
            f"Most confident misclassifications - {prefix} / {split.upper()}  |  "
            f"{wrong.size:,} errors of {len(targets):,} ({error_rate*100:.2f}%)",
            fontsize=13, fontweight='bold'
        )
        # h_pad keeps the two-line titles off the images in the row above:
        # with axis('off') tight_layout does not reserve room for them.
        fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.0)

        print(f"\nMisclassified examples plot [{prefix} | split={split.upper()}] - "
              f"showing {n} of {wrong.size:,} errors")
        return save_figure(fig, save_dir, f"misclassified_{prefix}_{split}",
                           plot_format, dpi, show)

    # ------------------------------------------------------------------
    # Plot everything
    # ------------------------------------------------------------------

    def plot_all(
        self,
        results: Dict,
        save_dir: Optional[str] = None,
        show: bool = False,
        dpi: int = 150,
        plot_format: str = 'pdf',
        top_n_misclassified: int = 16,
    ) -> Dict[str, Any]:
        """
        Produce every evaluation plot for one split.  Each plot failure is
        reported but does not abort the rest.
        """
        written: Dict[str, Any] = {}
        tasks = [
            ('curves', lambda: self.plot_curves(results, save_dir, show, dpi, plot_format)),
            ('confusion_matrix', lambda: self.plot_confusion_matrix(
                results, save_dir, show, dpi, plot_format)),
            ('per_class_metrics', lambda: self.plot_per_class_metrics(
                results, save_dir, show, dpi, plot_format)),
            ('class_distribution', lambda: self.plot_class_distribution(
                results, save_dir, show, dpi, plot_format)),
            ('confidence_hist', lambda: self.plot_confidence_histogram(
                results, save_dir, show, dpi, plot_format)),
            ('calibration', lambda: self.plot_calibration(
                results, save_dir, show, dpi, plot_format)),
            ('misclassified', lambda: self.plot_misclassified(
                results, save_dir, show, dpi, plot_format, top_n_misclassified)),
        ]
        for name, task in tasks:
            try:
                written[name] = task()
            except Exception as exc:
                print(f"{WARN} Plot '{name}' failed: {exc}")
        return written

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def _json_safe(value):
        """Recursively convert numpy scalars/arrays into JSON-friendly types."""
        if isinstance(value, dict):
            return {k: BaseEvaluator._json_safe(v) for k, v in value.items()
                    if not str(k).startswith('_')}
        if isinstance(value, (list, tuple)):
            return [BaseEvaluator._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        # json.dump would write NaN / Infinity, which is not valid JSON.
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def _write_json(self, results: Dict, path: str) -> str:
        payload = {k: self._json_safe(v) for k, v in results.items()
                   if not k.startswith('_')}
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2)
        print(f"  {OK} saved {ARROW} {path}")
        return path

    def export_results(
        self,
        results: Dict,
        out_dir: str,
        save_predictions: bool = False,
    ) -> Dict[str, str]:
        """
        Write the metrics of one split to disk.

        Produces:
            metrics_<model>_<split>.json               every scalar + CM + per-class
            per_class_metrics_<model>_<split>.csv      one row per class
            classification_report_<model>_<split>.txt  sklearn text report
            predictions_<model>_<split>.csv            per-sample predictions (opt-in)
        """
        split = results.get('split', 'split')
        prefix = results.get('model', 'model')
        stem = f"{prefix}_{split}"
        class_names = results.get('class_names') or self._class_names(split)
        os.makedirs(out_dir, exist_ok=True)
        written: Dict[str, str] = {}

        print(f"\nExporting results [{prefix} | split={split.upper()}] {ARROW} {out_dir}")

        written['json'] = self._write_json(
            results, os.path.join(out_dir, f"metrics_{stem}.json"))

        # Per-class CSV
        per_class = results['per_class']
        columns = ['precision', 'recall', 'f1', 'specificity', 'accuracy',
                   'support', 'predicted_count', 'true_positives',
                   'false_positives', 'false_negatives', 'mean_confidence',
                   'roc_auc', 'ap']
        path = os.path.join(out_dir, f"per_class_metrics_{stem}.csv")
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['class'] + columns)
            for name in class_names:
                stats = per_class.get(name)
                if stats is None:
                    continue
                writer.writerow([name] + [stats.get(c, '') for c in columns])
        print(f"  {OK} saved {ARROW} {path}")
        written['per_class_csv'] = path

        # Text classification report
        path = os.path.join(out_dir, f"classification_report_{stem}.txt")
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(f"Model: {prefix}\n")
            fh.write(f"Split: {split}\n")
            fh.write(f"Checkpoint: {results.get('model_path')}\n")
            fh.write(f"Samples: {results['n_samples']}\n")
            if results.get('note'):
                fh.write(f"NOTE: {results['note']}\n")
            fh.write("\n")
            fh.write(classification_report(
                results['_targets'], results['_predictions'],
                labels=list(range(self.num_classes)),
                target_names=class_names, zero_division=0, digits=4))
            fh.write("\nConfusion matrix (rows = true, cols = predicted):\n")
            fh.write(f"{class_names}\n")
            fh.write(np.array2string(results['_confusion_matrix'], max_line_width=200))
            fh.write("\n")
        print(f"  {OK} saved {ARROW} {path}")
        written['report_txt'] = path

        # Per-sample predictions
        if save_predictions:
            path = os.path.join(out_dir, f"predictions_{stem}.csv")
            probabilities = results['_probabilities']
            has_probs = probabilities is not None and len(probabilities)
            rel_paths = results.get('_rel_paths')
            with open(path, 'w', newline='', encoding='utf-8') as fh:
                writer = csv.writer(fh)
                header = ['index', 'target', 'target_name', 'prediction',
                          'prediction_name', 'correct', 'confidence']
                if rel_paths is not None:
                    header.insert(1, 'rel_path')
                if has_probs:
                    header += [f"prob_{name}" for name in class_names]
                writer.writerow(header)
                for i in range(len(results['_targets'])):
                    target = int(results['_targets'][i])
                    prediction = int(results['_predictions'][i])
                    row = [int(results['_indices'][i])]
                    if rel_paths is not None:
                        row.append(rel_paths[i])
                    row += [target, class_names[target],
                            prediction, class_names[prediction],
                            int(target == prediction),
                            f"{results['_confidences'][i]:.6f}"]
                    if has_probs:
                        row += [f"{p:.6f}" for p in probabilities[i]]
                    writer.writerow(row)
            print(f"  {OK} saved {ARROW} {path}")
            written['predictions_csv'] = path

        return written

    # ------------------------------------------------------------------
    # Complexity / latency reports
    # ------------------------------------------------------------------
    #
    # Every analysis section keeps the lines it printed under a `_report`
    # key: `_json_safe` strips it from the JSON, and the TXT export writes
    # exactly what was shown on the terminal.

    @staticmethod
    def _emit(lines: List[str]) -> List[str]:
        """Print a block of report lines and hand them back for the TXT export."""
        for line in lines:
            print(line)
        return lines

    @staticmethod
    def _timing_stats(times_ms: Sequence[float]) -> Dict[str, float]:
        """Summary statistics of a list of latencies in milliseconds."""
        t = np.asarray(times_ms, dtype=np.float64)
        return {
            'mean_ms': float(np.mean(t)),
            'std_ms': float(np.std(t)),
            'median_ms': float(np.median(t)),
            'min_ms': float(np.min(t)),
            'max_ms': float(np.max(t)),
            'p95_ms': float(np.percentile(t, 95)),
            'p99_ms': float(np.percentile(t, 99)),
        }

    @staticmethod
    def _format_count(num: Optional[float]) -> str:
        if num is None:
            return 'N/A'
        if num >= 1e9:
            return f"{num/1e9:.2f}G"
        if num >= 1e6:
            return f"{num/1e6:.2f}M"
        if num >= 1e3:
            return f"{num/1e3:.2f}K"
        return str(num)

    def _write_report(self, sections: Sequence[Optional[Dict]], header: str,
                      path: str) -> str:
        """Write the `_report` lines of every section into a text file."""
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(header + "\n\n")
            for section in sections:
                if section and section.get('_report'):
                    fh.write("\n".join(section['_report']) + "\n\n")
        print(f"  {OK} saved {ARROW} {path}")
        return path

    def _report_header(self, title: str) -> str:
        return (f"{title}\n"
                f"Model: {getattr(self, 'MODEL_NAME', 'model')}\n"
                f"Checkpoint: {self.model_path}\n"
                f"Generated: {_dt.datetime.now().isoformat(timespec='seconds')}")

    def export_model_summary(self, summary: Dict, out_dir: str) -> Dict[str, str]:
        """
        Write the model analysis (complexity, benchmark, memory, pipeline).

        Produces:
            model_summary_<model>.json   every value, machine readable
            model_summary_<model>.txt    the report printed on the terminal

        The complexity does not depend on the split, so the name carries none.
        The JSON is also what the stacking run reads back to aggregate the
        JITrans / Mamba numbers into the full-pipeline figures.
        """
        model = getattr(self, 'MODEL_NAME', 'model')
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nExporting model analysis [{model}] {ARROW} {out_dir}")

        payload = {'kind': 'model_summary', 'model': model,
                   'checkpoint': self.model_path,
                   'generated': _dt.datetime.now().isoformat(timespec='seconds')}
        payload.update(summary)
        written = {
            'json': self._write_json(
                payload, os.path.join(out_dir, f"model_summary_{model}.json")),
            'txt': self._write_report(
                [summary.get(k) for k in ('complexity', 'pipeline', 'benchmark', 'memory')],
                self._report_header("MODEL ANALYSIS"),
                os.path.join(out_dir, f"model_summary_{model}.txt")),
        }
        return written

    def export_single_inference(self, summary: Dict, out_dir: str) -> Dict[str, str]:
        """
        Write the single-image inference timings.

        Produces:
            single_inference_<model>.json   latency statistics + metadata
            single_inference_<model>.txt    the report printed on the terminal
            single_inference_<model>.csv    one row per processed image
        """
        model = getattr(self, 'MODEL_NAME', 'model')
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nExporting single-image inference [{model}] {ARROW} {out_dir}")

        payload = {'kind': 'single_inference', 'model': model,
                   'checkpoint': self.model_path,
                   'generated': _dt.datetime.now().isoformat(timespec='seconds')}
        payload.update(summary)
        written = {
            'json': self._write_json(
                payload, os.path.join(out_dir, f"single_inference_{model}.json")),
            'txt': self._write_report(
                [summary], self._report_header("SINGLE-IMAGE INFERENCE"),
                os.path.join(out_dir, f"single_inference_{model}.txt")),
        }

        rows = summary.get('_rows') or []
        if rows:
            path = os.path.join(out_dir, f"single_inference_{model}.csv")
            with open(path, 'w', newline='', encoding='utf-8') as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                                     for k, v in row.items()})
            print(f"  {OK} saved {ARROW} {path}")
            written['csv'] = path
        return written

    def _sample_indices(self, n_available: int, num_samples: int,
                        seed: Optional[int]) -> np.ndarray:
        """Random, reproducible pick of the images used for the latency run."""
        rng = np.random.default_rng(seed)
        return rng.choice(n_available, size=min(num_samples, n_available),
                          replace=False)

    def _device_name(self) -> str:
        """GPU model, or the CPU description for CPU / scikit-learn runs."""
        device = getattr(self, 'device', None)
        if device is not None and device.type == 'cuda':
            return torch.cuda.get_device_name(device)
        import platform
        return platform.processor() or platform.machine() or 'CPU'

    _PHASE_LABELS = {
        'load_ms': 'Load + preprocess',
        'h2d_ms': 'Host -> device',
        'forward_ms': 'Forward pass',
        'predict_ms': 'predict_proba',
        'post_ms': 'Post-process',
        'model_ms': 'Model (no load)',
        'total_ms': 'Total per image',
    }

    def _single_inference_lines(self, summary: Dict,
                                extra: Optional[List[str]] = None) -> List[str]:
        """Terminal / TXT report of a `single_image_inference` summary."""
        lines = [
            "=" * 60,
            "SINGLE-IMAGE INFERENCE (batch = 1)",
            "=" * 60,
            f"Model:      {getattr(self, 'MODEL_NAME', 'model')}",
            f"Split:      {summary['split']}",
            f"Images:     {summary['num_samples']}  (warm-up {summary['warmup_runs']} runs "
            f"excluded, seed {summary['seed']})",
            f"Device:     {summary['device']} ({summary['device_name']})",
            f"Precision:  {summary['precision']}",
            "",
            f"{'Phase':<20}{'mean':>9}{'std':>9}{'median':>9}{'p95':>9}{'max':>9}   [ms]",
        ]
        for key, stats in summary['timing'].items():
            lines.append(
                f"{self._PHASE_LABELS.get(key, key):<20}{stats['mean_ms']:>9.3f}"
                f"{stats['std_ms']:>9.3f}{stats['median_ms']:>9.3f}"
                f"{stats['p95_ms']:>9.3f}{stats['max_ms']:>9.3f}")
        lines.append("")
        throughput = summary.get('throughput_imgs_per_sec')
        lines.append(f"Latency per image:  {summary['latency_ms']:.3f} ms"
                     + (f"  ->  {throughput:.2f} images/sec" if throughput else ''))
        memory = summary.get('memory')
        if memory and memory.get('peak_mb') is not None:
            overhead = memory.get('overhead_mb')
            lines.append(f"Memory ({memory['kind']}):  peak {memory['peak_mb']:.2f} MB"
                         + (f", overhead {overhead:.2f} MB" if overhead is not None else ''))
        if summary.get('sample_accuracy') is not None:
            lines.append(f"Accuracy on these images: {summary['sample_accuracy']:.4f} "
                         f"(sanity check only)")
        if extra:
            lines += [""] + extra
        lines.append("=" * 60)
        return lines


# ======================================================================
# JITrans inference
# ======================================================================

class JITInference(BaseEvaluator):
    """
    Evaluation of the JITrans classifier (`model.Transformer` via
    `model.define_model`) on the RSCD dataset.

    Split options:
        split='val'      -> validation dataset only (default)
        split='test'     -> test dataset only
        split='both'     -> val + test concatenated into a single dataset
        split='separate' -> val AND test, evaluated and reported independently
    """

    MODEL_NAME = 'jit'

    def __init__(
        self,
        model_path: str,
        data_path: str,
        data_set: str = 'RSCD',
        img_size: int = 224,
        patch_size: Optional[int] = None,
        feedback_steps: Optional[int] = None,
        eval_crop_ratio: float = 0.875,
        batch_size: int = 64,
        num_workers: int = 4,
        device: Optional[torch.device] = None,
        split: str = 'val',
        class_names: Optional[Sequence[str]] = None,
        amp: bool = False,
        strict: bool = False,
        use_checkpoint: bool = False,
        **kwargs
    ):
        require_eval_deps()
        if split not in ('val', 'test', 'both', 'separate'):
            raise ValueError(
                f"split must be 'val', 'test', 'both' or 'separate', got '{split}'"
            )

        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.eval_crop_ratio = eval_crop_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.amp = amp
        self.strict = strict
        self.use_checkpoint = use_checkpoint

        # The checkpoint stores the training args (main.py:338); patch_size and
        # feedback_steps MUST match or the weights will not load, so they are
        # taken from there unless explicitly overridden on the CLI.
        checkpoint = self._read_checkpoint(model_path)
        train_args = checkpoint.get('args') if isinstance(checkpoint, dict) else None
        train_args = train_args if isinstance(train_args, dict) else {}

        self.patch_size = self._resolve_hparam(
            'patch_size', patch_size, train_args, default=16)
        self.feedback_steps = self._resolve_hparam(
            'feedback_steps', feedback_steps, train_args, default=4)
        if train_args.get('img_size') and img_size != train_args['img_size']:
            print(f"{WARN} --img-size {img_size} differs from the checkpoint's "
                  f"{train_args['img_size']}")

        # Datasets
        self.datasets, num_classes = self._create_datasets(
            data_path, data_set, split, **kwargs)

        ckpt_class_names = checkpoint.get('class_names') if isinstance(checkpoint, dict) else None
        resolved_names = (list(class_names) if class_names
                          else list(ckpt_class_names) if ckpt_class_names
                          else None)

        super().__init__(
            num_classes=num_classes,
            split_names=list(self.datasets.keys()),
            class_names=resolved_names,
            model_path=model_path,
            split=split,
        )

        self.dataloaders = {
            name: DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,          # position == dataset index
                num_workers=num_workers,
                pin_memory=self.device.type == 'cuda',
                drop_last=False,
            )
            for name, dataset in self.datasets.items()
        }

        self.model = self._load_model(checkpoint)

        print(f"\n{OK} JITrans inference setup complete")
        print(f"  Device:         {self.device}")
        print(f"  Split mode:     {self.split.upper()}")
        for name in self.split_names:
            print(f"    - {name:<10} {len(self.datasets[name])} samples")
        print(f"  Batch size:     {self.batch_size}")
        print(f"  Image size:     {self.img_size}  patch {self.patch_size}")
        print(f"  Feedback steps: {self.feedback_steps}")
        print(f"  Num classes:    {self.num_classes}")
        print(f"  Class names:    {', '.join(self._class_names())}")
        print(f"  AMP:            {self.amp}")

    # -- helpers ------------------------------------------------------

    @staticmethod
    def _read_checkpoint(path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location='cpu', weights_only=False)

    @staticmethod
    def _resolve_hparam(name, cli_value, train_args: Dict, default):
        """CLI wins, then the value stored in the checkpoint, then the default."""
        if cli_value is not None:
            if name in train_args and train_args[name] != cli_value:
                print(f"{WARN} --{name.replace('_', '-')} {cli_value} overrides the "
                      f"checkpoint value {train_args[name]}; weights may fail to load")
            return cli_value
        if name in train_args and train_args[name] is not None:
            print(f"  {name} = {train_args[name]}  (from the checkpoint args)")
            return train_args[name]
        return default

    def _class_names(self, split: Optional[str] = None) -> List[str]:
        if self._class_names_override:
            return list(self._class_names_override)
        key = split if split in self.datasets else self.split_names[0]
        names = getattr(self.datasets[key], 'classes', None)
        if names:
            return list(names)
        return [f"class_{i}" for i in range(self.num_classes)]

    # -- dataset ------------------------------------------------------

    def _create_datasets(
        self,
        data_path: str,
        data_set: str,
        split: str,
        **kwargs
    ) -> Tuple[Dict[str, Any], int]:
        """Build the requested split(s) through the repo's own `build_dataset`."""
        from dataloader import build_dataset

        class Args:
            pass

        args = Args()
        args.data_path = data_path
        args.data_set = data_set
        args.img_size = self.img_size
        args.patch_size = self.patch_size
        args.eval_crop_ratio = self.eval_crop_ratio
        args.color_jitter = kwargs.get('color_jitter', 0.4)
        args.aa = kwargs.get('aa', 'rand-m9-mstd0.5-inc1')
        args.train_interpolation = kwargs.get('train_interpolation', 'bicubic')
        args.reprob = kwargs.get('reprob', 0.25)
        args.remode = kwargs.get('remode', 'pixel')
        args.recount = kwargs.get('recount', 1)

        def _build(is_test: bool, label: str):
            dataset, num_classes = build_dataset(
                is_train=False, is_test=is_test, args=args)
            print(f"{OK} {label} dataset created: {data_set} | "
                  f"{len(dataset)} samples | {num_classes} classes")
            return dataset, num_classes

        try:
            if split == 'val':
                dataset, num_classes = _build(False, 'Validation')
                return {'val': dataset}, num_classes

            if split == 'test':
                dataset, num_classes = _build(True, 'Test')
                return {'test': dataset}, num_classes

            dataset_val, num_classes = _build(False, 'Validation')
            dataset_test, num_classes_test = _build(True, 'Test')

            if num_classes != num_classes_test:
                raise RuntimeError(
                    f"Val and test datasets have different num_classes: "
                    f"{num_classes} vs {num_classes_test}"
                )

            if split == 'separate':
                print(f"{OK} Separate evaluation mode: val ({len(dataset_val)}) and "
                      f"test ({len(dataset_test)}) will be reported independently")
                return {'val': dataset_val, 'test': dataset_test}, num_classes

            dataset = ConcatDataset([dataset_val, dataset_test])
            # ConcatDataset has no .classes; propagate it from the first dataset
            if hasattr(dataset_val, 'classes'):
                dataset.classes = dataset_val.classes
            print(
                f"{OK} Combined (val+test) dataset created: {data_set} | "
                f"{len(dataset_val)} val + {len(dataset_test)} test = "
                f"{len(dataset)} total samples | {num_classes} classes"
            )
            return {'both': dataset}, num_classes

        except Exception as exc:
            raise RuntimeError(f"Failed to create dataset (split='{split}'): {exc}")

    # -- model --------------------------------------------------------

    def _load_model(self, checkpoint) -> torch.nn.Module:
        """
        Instantiate JITrans and load the checkpoint weights.

        Accepts every format `main.py` / older scripts may have written:
        a dict with `model_state_dict` (main.py:334), a dict with `state_dict`,
        a bare state dict, or a directly pickled module.  DDP checkpoints carry
        a `module.` prefix that is stripped here.
        """
        from model import define_model

        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint.to(self.device)
            print(f"{OK} Model loaded as a pickled module: {self.model_path}")
        else:
            model = define_model(
                self.device,
                num_classes=self.num_classes,
                patch_size=self.patch_size,
                img_size=self.img_size,
                feedback_steps=self.feedback_steps,
                use_checkpoint=self.use_checkpoint,
            )

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state = checkpoint['state_dict']
            else:
                state = checkpoint

            if any(k.startswith('module.') for k in state):
                state = {k[len('module.'):] if k.startswith('module.') else k: v
                         for k, v in state.items()}

            missing, unexpected = model.load_state_dict(state, strict=self.strict)
            if missing:
                print(f"{WARN} missing keys ({len(missing)}): {list(missing)[:6]}")
            if unexpected:
                print(f"{WARN} unexpected keys ({len(unexpected)}): {list(unexpected)[:6]}")

            print(f"{OK} Model loaded from checkpoint: {self.model_path}")
            if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                # main.py stores a 0-based epoch index
                print(f"  epoch = {checkpoint.get('epoch')}  "
                      f"best_acc = {checkpoint.get('best_acc')}")

        model.eval()
        model = model.to(self.device)
        print(f"{OK} Model is on device: {next(model.parameters()).device}")
        return model

    # -- hooks --------------------------------------------------------

    def _get_image(self, split_name: str, index: int) -> torch.Tensor:
        return self.datasets[split_name][index][0]

    def _predict_split(
        self, split_name: str, verbose: bool = True, track_performance: bool = True
    ):
        """Forward pass over one split, with timing and memory tracking."""
        dataset = self.datasets[split_name]
        dataloader = self.dataloaders[split_name]

        all_probabilities: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        batch_times: List[float] = []
        memory_samples: List[float] = []

        total_images = len(dataset)
        processed_images = 0
        process = None

        if track_performance:
            if self.device.type == 'cuda':
                # Synchronize so all pending ops are done, then read the current
                # footprint (model weights already live in the pool).  Do NOT
                # call empty_cache() - that would evict the model weights and
                # give a falsely-low baseline.
                torch.cuda.synchronize()
                initial_allocated_mb = torch.cuda.memory_allocated() / 1024**2
                initial_reserved_mb = torch.cuda.memory_reserved() / 1024**2
                # Reset the peak counter AFTER reading the baseline so the peak
                # reflects inference overhead on top of the loaded model.
                torch.cuda.reset_peak_memory_stats()
            elif PSUTIL_AVAILABLE:
                process = psutil.Process(os.getpid())
                initial_allocated_mb = process.memory_info().rss / 1024**2
                initial_reserved_mb = None
            else:
                initial_allocated_mb = None
                initial_reserved_mb = None

        if verbose:
            print(f"Starting evaluation on {total_images} images "
                  f"(split={split_name.upper()})...")
            print(f"Batch size: {self.batch_size}")
            if track_performance and self.device.type == 'cuda':
                print(f"GPU memory baseline - allocated: {initial_allocated_mb:.1f} MB  "
                      f"reserved: {initial_reserved_mb:.1f} MB")

        start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if track_performance:
                    batch_start = time.perf_counter()

                inputs = inputs.to(self.device, non_blocking=True)
                with self._autocast():
                    outputs = self.model(inputs)
                # `return_all_steps=False` gives a plain tensor, but stay
                # tolerant in case a caller wired the tuple form.
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                probs = torch.softmax(outputs.float(), dim=1)

                if track_performance:
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    batch_times.append((time.perf_counter() - batch_start) * 1000)
                    if self.device.type == 'cuda':
                        memory_samples.append(torch.cuda.memory_allocated() / 1024**2)
                    elif process is not None:
                        memory_samples.append(process.memory_info().rss / 1024**2)

                all_probabilities.append(probs.cpu().numpy())
                all_targets.append(targets.numpy())
                processed_images += len(inputs)

                if verbose and (batch_idx + 1) % max(1, len(dataloader) // 10) == 0:
                    elapsed_time = time.time() - start_time
                    progress = processed_images / total_images * 100
                    eta = elapsed_time / processed_images * (total_images - processed_images)
                    throughput = processed_images / elapsed_time
                    print(f"Progress: {progress:.1f}% ({processed_images}/{total_images}) - "
                          f"ETA: {eta:.1f}s - Throughput: {throughput:.2f} imgs/s")

        probabilities = np.concatenate(all_probabilities, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        # shuffle=False, drop_last=False -> running position == dataset index,
        # which is what lets `plot_misclassified` fetch the images back.
        indices = np.arange(len(targets))

        performance = None
        if track_performance and batch_times:
            total_time = time.time() - start_time
            t_arr = np.array(batch_times)
            m_arr = np.array(memory_samples) if memory_samples else None
            mean_time_per_image = float(np.mean(t_arr)) / self.batch_size

            performance = {
                'device': 'GPU (CUDA)' if self.device.type == 'cuda' else 'CPU',
                'total_time_s': total_time,
                'total_images': total_images,
                'batch_size': self.batch_size,
                'num_batches': len(batch_times),
                'mean_batch_time_ms': float(np.mean(t_arr)),
                'std_batch_time_ms': float(np.std(t_arr)),
                'median_batch_time_ms': float(np.median(t_arr)),
                'mean_time_per_image_ms': mean_time_per_image,
                'throughput_imgs_per_sec': total_images / total_time,
            }

            if self.device.type == 'cuda':
                # Peak allocated AFTER the baseline reset = pure inference overhead
                peak_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
                performance.update({
                    'baseline_allocated_mb': initial_allocated_mb,
                    'baseline_reserved_mb': initial_reserved_mb,
                    'peak_allocated_mb': peak_allocated_mb,
                    'peak_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2,
                    'avg_allocated_mb': float(np.mean(m_arr)) if m_arr is not None else None,
                    'inference_overhead_mb': peak_allocated_mb - initial_allocated_mb,
                    'memory_per_image_mb': max(peak_allocated_mb - initial_allocated_mb, 0) / total_images,
                })
            elif m_arr is not None and initial_allocated_mb is not None:
                peak_rss_mb = float(np.max(m_arr))
                performance.update({
                    'baseline_allocated_mb': initial_allocated_mb,
                    'peak_allocated_mb': peak_rss_mb,
                    'avg_allocated_mb': float(np.mean(m_arr)),
                    'inference_overhead_mb': peak_rss_mb - initial_allocated_mb,
                    'memory_per_image_mb': max(peak_rss_mb - initial_allocated_mb, 0) / total_images,
                })

        return probabilities, targets, indices, performance

    # ------------------------------------------------------------------
    # Complexity / benchmark
    # ------------------------------------------------------------------

    def _autocast(self):
        """The same autocast context used by the evaluation forward pass."""
        return torch.amp.autocast('cuda', enabled=self.amp and self.device.type == 'cuda')

    @property
    def _precision(self) -> str:
        return 'amp (fp16 autocast)' if self.amp and self.device.type == 'cuda' else 'fp32'

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _flop_breakdown(flop_counter, total_flops: int) -> Optional[Dict[str, Dict]]:
        """FLOPs per top-level sub-module (encoder, decoder, embedding, head...)."""
        try:
            counts = flop_counter.get_flop_counts()
        except Exception:
            return None
        breakdown = {}
        # Keys are fully qualified module names rooted at the model class
        # ("Transformer.enc", "Transformer.dec.layers.0", ...): one dot = a
        # direct child.  Modules called several times (the decoder, once per
        # feedback step) accumulate over every call.
        for name, ops in counts.items():
            if name.count('.') != 1:
                continue
            flops = int(sum(ops.values()))
            if flops:
                breakdown[name.split('.', 1)[1]] = {
                    'flops': flops,
                    'share': flops / total_flops if total_flops else None,
                }
        return breakdown or None

    def calculate_model_complexity(
        self,
        input_size: Optional[Tuple[int, int, int, int]] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Model complexity: parameter counts, size in MB and FLOPs.

        FLOPs come from torch's FlopCounterMode, which counts 2 FLOPs per
        multiply-accumulate; `macs` = flops / 2 is the figure that fvcore /
        thop (and most papers) report as "FLOPs".  JITrans is built only from
        Linear and matmul ops, all visible to the counter; softmax, LayerNorm
        and the activations are not counted (negligible).  The count runs in
        FP32: autocast changes the precision, not the number of operations.
        """
        input_size = input_size or (1, 3, self.img_size, self.img_size)
        fmt = self._format_count

        params = list(self.model.parameters())
        total_params = sum(p.numel() for p in params)
        trainable_params = sum(p.numel() for p in params if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        param_size = sum(p.nelement() * p.element_size() for p in params)
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        checkpoint_size_mb = (os.path.getsize(self.model_path) / 1024**2
                              if self.model_path and os.path.isfile(self.model_path)
                              else None)
        grid = self.img_size // self.patch_size

        results = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'total_params_formatted': fmt(total_params),
            'trainable_params_formatted': fmt(trainable_params),
            'model_size_mb': model_size_mb,
            'checkpoint_size_mb': checkpoint_size_mb,
            'flops': None,
            'flops_formatted': 'N/A',
            'macs': None,
            'macs_formatted': 'N/A',
            'gflops': None,
            'gmacs': None,
            'flops_convention': 'FLOPs = 2 x MACs',
            'flops_breakdown': None,
            'method': 'manual',
            'input_size': list(input_size),
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'num_tokens': grid * grid,
            'feedback_steps': self.feedback_steps,
            'device': str(self.device),
            'model_type': type(self.model).__name__,
            'torch_version': torch.__version__,
            'notes': [
                'model_size_mb = parameters + registered buffers in memory; '
                'PositionalEncoding.pe is a plain tensor attribute (not a buffer) '
                'and is not included',
            ],
        }

        if FLOP_COUNTER_AVAILABLE:
            try:
                dummy_input = torch.randn(*input_size, device=self.device)
                self.model.eval()
                flop_counter = FlopCounterMode(display=False)
                # Grad mode must stay ON: under no_grad / inference_mode the
                # counter's module tracker fails ("Expected gradient function
                # to be set") and the FLOPs silently end up N/A.  Only a
                # forward pass runs, nothing is back-propagated.
                with torch.enable_grad(), flop_counter:
                    _ = self.model(dummy_input)
                total_flops = int(flop_counter.get_total_flops())
                results.update({
                    'flops': total_flops,
                    'flops_formatted': fmt(total_flops),
                    'macs': total_flops // 2,
                    'macs_formatted': fmt(total_flops // 2),
                    'gflops': total_flops / 1e9,
                    'gmacs': total_flops / 2e9,
                    'flops_breakdown': self._flop_breakdown(flop_counter, total_flops),
                    'method': 'torch_flop_counter',
                })
            except Exception as exc:
                results['flops_error'] = str(exc)
                if verbose:
                    print(f"{WARN} Could not calculate FLOPs: {exc}")

        size_ckpt = (f"{checkpoint_size_mb:.2f} MB" if checkpoint_size_mb is not None
                     else 'N/A')
        lines = [
            "=" * 60,
            "MODEL COMPLEXITY ANALYSIS",
            "=" * 60,
            f"Total Parameters:      {results['total_params_formatted']} ({total_params:,})",
            f"Trainable Parameters:  {results['trainable_params_formatted']} ({trainable_params:,})",
            f"Non-trainable Params:  {fmt(non_trainable_params)} ({non_trainable_params:,})",
            f"Model Size (memory):   {model_size_mb:.2f} MB",
            f"Checkpoint on disk:    {size_ckpt}",
            f"FLOPs:                 {results['flops_formatted']}"
            + (f"  ({results['gflops']:.3f} GFLOPs)" if results['gflops'] is not None else ''),
            f"MACs:                  {results['macs_formatted']}"
            + (f"  ({results['gmacs']:.3f} GMACs)" if results['gmacs'] is not None else ''),
            f"Convention:            {results['flops_convention']}",
            f"Input Size:            {tuple(input_size)}",
            f"Tokens:                {grid * grid} ({grid}x{grid}, patch {self.patch_size})",
            f"Feedback steps:        {self.feedback_steps}",
            f"Device:                {self.device}",
            f"Model Type:            {results['model_type']}",
            f"Method:                {results['method']}",
        ]
        if results['flops_breakdown']:
            lines.append("FLOPs per sub-module:")
            for name, entry in results['flops_breakdown'].items():
                lines.append(f"  {name:<18} {fmt(entry['flops']):>9}  "
                             f"({entry['share'] * 100:5.1f}%)")
        lines.append("=" * 60)
        results['_report'] = self._emit(lines) if verbose else lines
        return results

    def benchmark_inference(
        self,
        input_size: Optional[Tuple[int, int, int, int]] = None,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict:
        """
        Benchmark raw forward-pass latency on one synthetic input.

        Runs under the same autocast as the evaluation (`--amp`), so the
        number is comparable with the evaluation / single-image latencies.
        """
        input_size = input_size or (1, 3, self.img_size, self.img_size)
        dummy_input = torch.randn(*input_size).to(self.device)
        self.model.eval()

        print(f"Warming up with {warmup_runs} runs...")
        with torch.no_grad():
            for _ in range(warmup_runs):
                with self._autocast():
                    _ = self.model(dummy_input)
        self._sync()

        print(f"Running {num_runs} inference benchmarks...")
        times = []
        with torch.no_grad():
            for i in range(num_runs):
                start_time = time.perf_counter()
                with self._autocast():
                    _ = self.model(dummy_input)
                self._sync()
                times.append((time.perf_counter() - start_time) * 1000)
                if (i + 1) % 20 == 0:
                    print(f"Completed {i + 1}/{num_runs} runs")

        times = np.array(times)
        results = {
            'mean_time_ms': float(np.mean(times)),
            'std_time_ms': float(np.std(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times)),
            'median_time_ms': float(np.median(times)),
            'p95_time_ms': float(np.percentile(times, 95)),
            'p99_time_ms': float(np.percentile(times, 99)),
            'fps': 1000.0 / float(np.mean(times)),
            'throughput': (input_size[0] * 1000.0) / float(np.mean(times)),
            'input_size': list(input_size),
            'device': str(self.device),
            'device_name': self._device_name(),
            'precision': self._precision,
            'num_runs': num_runs,
            'warmup_runs': warmup_runs,
        }

        lines = [
            "=" * 60,
            "INFERENCE BENCHMARK RESULTS (synthetic input, forward only)",
            "=" * 60,
            f"Mean Time:    {results['mean_time_ms']:.2f} +/- {results['std_time_ms']:.2f} ms",
            f"Median Time:  {results['median_time_ms']:.2f} ms",
            f"P95 / P99:    {results['p95_time_ms']:.2f} / {results['p99_time_ms']:.2f} ms",
            f"Min/Max Time: {results['min_time_ms']:.2f} / {results['max_time_ms']:.2f} ms",
            f"FPS (batch={input_size[0]}): {results['fps']:.2f}",
            f"Throughput:   {results['throughput']:.2f} images/sec",
            f"Device:       {self.device} ({results['device_name']})",
            f"Precision:    {self._precision}",
            "=" * 60,
        ]
        print()
        results['_report'] = self._emit(lines)
        return results

    def memory_benchmark(
        self,
        input_size: Optional[Tuple[int, int, int, int]] = None,
        num_runs: int = 10
    ) -> Dict:
        """
        Benchmark memory usage during inference.

        NOTES on CUDA memory tracking:
          - memory_allocated() returns memory used by live tensors.  After the
            model is loaded the baseline is non-zero, so it is subtracted and
            the deltas represent inference overhead only.
          - memory_reserved() is the total pool size claimed from the OS;
            it is never smaller than memory_allocated().
          - memory_efficiency = allocated / reserved * 100.  When both are 0 we
            report 'N/A' instead of NaN to avoid a RuntimeWarning.
        """
        input_size = input_size or (1, 3, self.img_size, self.img_size)
        process = None

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            initial_allocated = torch.cuda.memory_allocated() / 1024**2
            initial_reserved = torch.cuda.memory_reserved() / 1024**2
            torch.cuda.reset_peak_memory_stats()
            print("Starting GPU memory benchmark...")
            print(f"  Baseline allocated (model loaded): {initial_allocated:.2f} MB")
            print(f"  Baseline reserved  (pool size):    {initial_reserved:.2f} MB")
        else:
            if not PSUTIL_AVAILABLE:
                lines = [f"{WARN} psutil is not installed - CPU memory benchmark skipped"]
                return {'device': 'CPU', 'error': 'psutil not available',
                        '_report': self._emit(lines)}
            process = psutil.Process(os.getpid())
            initial_allocated = process.memory_info().rss / 1024**2
            initial_reserved = None
            print("Starting CPU memory benchmark...")
            print(f"  Initial RSS: {initial_allocated:.2f} MB")

        dummy_input = torch.randn(*input_size).to(self.device)
        allocated_samples, reserved_samples, peak_samples = [], [], []

        print(f"Running {num_runs} memory profiling runs...")
        with torch.no_grad():
            for i in range(num_runs):
                if self.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()

                with self._autocast():
                    _ = self.model(dummy_input)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    allocated_samples.append(torch.cuda.memory_allocated() / 1024**2)
                    reserved_samples.append(torch.cuda.memory_reserved() / 1024**2)
                    peak_samples.append(torch.cuda.max_memory_allocated() / 1024**2)
                else:
                    allocated_samples.append(process.memory_info().rss / 1024**2)

                if (i + 1) % 5 == 0:
                    print(f"  Completed {i + 1}/{num_runs} runs")

        if self.device.type == 'cuda':
            avg_allocated = float(np.mean(allocated_samples))
            avg_reserved = float(np.mean(reserved_samples))
            max_peak = float(np.max(peak_samples))

            if avg_reserved > 0:
                efficiency = (avg_allocated / avg_reserved) * 100.0
                efficiency_str = f"{efficiency:.1f}%"
            else:
                efficiency = float('nan')
                efficiency_str = "N/A (reserved=0, pool may be pre-allocated)"

            results = {
                'device': 'GPU (CUDA)',
                'device_name': self._device_name(),
                'precision': self._precision,
                'input_size': list(input_size),
                'num_runs': num_runs,
                'baseline_allocated_mb': initial_allocated,
                'baseline_reserved_mb': initial_reserved,
                'avg_allocated_mb': avg_allocated,
                'max_allocated_mb': float(np.max(allocated_samples)),
                'avg_reserved_mb': avg_reserved,
                'max_reserved_mb': float(np.max(reserved_samples)),
                'avg_peak_mb': float(np.mean(peak_samples)),
                'max_peak_mb': max_peak,
                'inference_overhead_mb': max_peak - initial_allocated,
                'memory_efficiency': efficiency,
                'memory_efficiency_str': efficiency_str,
            }

            lines = [
                "=" * 60,
                "GPU MEMORY USAGE RESULTS",
                "=" * 60,
                f"Baseline (model loaded):   {initial_allocated:.2f} MB allocated  /  "
                f"{initial_reserved:.2f} MB reserved",
                "During inference:",
                f"  Avg allocated:           {avg_allocated:.2f} MB",
                f"  Peak allocated:          {results['max_allocated_mb']:.2f} MB",
                f"  Avg reserved (pool):     {avg_reserved:.2f} MB",
                f"  Peak during inference:   {max_peak:.2f} MB",
                f"Inference overhead:        {results['inference_overhead_mb']:.2f} MB  "
                f"(peak - baseline)",
                f"Memory efficiency:         {efficiency_str}  (allocated / reserved)",
                f"Precision:                 {self._precision}",
                "=" * 60,
            ]
        else:
            avg_rss = float(np.mean(allocated_samples))
            max_rss = float(np.max(allocated_samples))
            results = {
                'device': 'CPU',
                'device_name': self._device_name(),
                'precision': self._precision,
                'input_size': list(input_size),
                'num_runs': num_runs,
                'baseline_allocated_mb': initial_allocated,
                'avg_rss_mb': avg_rss,
                'max_rss_mb': max_rss,
                'inference_overhead_mb': max_rss - initial_allocated,
            }
            lines = [
                "=" * 60,
                "CPU MEMORY USAGE RESULTS",
                "=" * 60,
                f"Initial RSS:    {initial_allocated:.2f} MB",
                f"Average RSS:    {avg_rss:.2f} MB",
                f"Peak RSS:       {max_rss:.2f} MB",
                f"Overhead:       {results['inference_overhead_mb']:.2f} MB",
                "=" * 60,
            ]

        print()
        results['_report'] = self._emit(lines)
        return results

    def model_summary(
        self,
        input_size: Optional[Tuple[int, int, int, int]] = None,
        include_benchmark: bool = False,
        include_memory: bool = False,
        benchmark_runs: int = 50,
        memory_runs: int = 10
    ) -> Dict:
        """Complexity, plus optionally the latency and memory benchmarks."""
        print("Starting Model Analysis...")
        print("=" * 60)

        results = {
            'complexity': self.calculate_model_complexity(input_size, verbose=True),
            'benchmark': None,
            'memory': None,
        }

        if include_benchmark:
            print("\n" + "=" * 60)
            results['benchmark'] = self.benchmark_inference(
                input_size, benchmark_runs, warmup_runs=10)

        if include_memory:
            print("\n" + "=" * 60)
            results['memory'] = self.memory_benchmark(input_size, memory_runs)

        print("\n" + "=" * 60)
        print(f"{OK} Model Analysis Complete!")
        print("=" * 60)
        return results

    # ------------------------------------------------------------------
    # Single-image inference latency
    # ------------------------------------------------------------------

    def single_image_inference(
        self,
        num_samples: int = 50,
        warmup_runs: int = 10,
        seed: Optional[int] = 42,
        split_name: Optional[str] = None,
    ) -> Dict:
        """
        Latency of one prediction at a time (batch = 1) on real images.

        Unlike `benchmark_inference` (one synthetic tensor, forward only) this
        times the whole path of a single prediction, split into phases:
            load_ms     dataset[idx]: image decode + eval transforms (CPU)
            h2d_ms      host -> device copy
            forward_ms  model forward pass (same autocast as the evaluation)
            post_ms     softmax + argmax + copy of the result to the host
            model_ms    h2d + forward + post: latency once the image is
                        already decoded and preprocessed
            total_ms    load + model_ms: end-to-end latency of one image
        The first `warmup_runs` forwards (CUDA context, cuDNN autotuning,
        allocator growth) are excluded; without them a run on few images
        mostly measures the cold start.
        """
        split_name = split_name or ('test' if 'test' in self.datasets
                                    else self.split_names[0])
        dataset = self.datasets[split_name]
        class_names = self._class_names(split_name)
        indices = self._sample_indices(len(dataset), num_samples, seed)
        on_cuda = self.device.type == 'cuda'
        self.model.eval()

        print(f"\nSingle-image inference: {len(indices)} images from "
              f"{split_name.upper()} (batch=1, warm-up {warmup_runs} runs, "
              f"{self._precision})")

        with torch.no_grad():
            warm_input = dataset[int(indices[0])][0].unsqueeze(0).to(self.device)
            for _ in range(warmup_runs):
                with self._autocast():
                    _ = self.model(warm_input)
            self._sync()
            del warm_input

        process = None
        if on_cuda:
            baseline_mb = torch.cuda.memory_allocated(self.device) / 1024**2
            torch.cuda.reset_peak_memory_stats(self.device)
        elif PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            baseline_mb = process.memory_info().rss / 1024**2
        else:
            baseline_mb = None
        peak_rss_mb = baseline_mb

        phases: Dict[str, List[float]] = {
            k: [] for k in ('load_ms', 'h2d_ms', 'forward_ms', 'post_ms',
                            'model_ms', 'total_ms')}
        rows: List[Dict[str, Any]] = []

        with torch.no_grad():
            for idx in indices:
                idx = int(idx)
                t0 = time.perf_counter()
                image, target = dataset[idx]
                t1 = time.perf_counter()
                x = image.unsqueeze(0).to(self.device)
                self._sync()
                t2 = time.perf_counter()
                with self._autocast():
                    outputs = self.model(x)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                self._sync()
                t3 = time.perf_counter()
                probs = torch.softmax(outputs.float(), dim=1)
                confidence, prediction = probs.max(dim=1)
                confidence, prediction = float(confidence.item()), int(prediction.item())
                t4 = time.perf_counter()

                if process is not None:
                    peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / 1024**2)

                times = {
                    'load_ms': (t1 - t0) * 1000,
                    'h2d_ms': (t2 - t1) * 1000,
                    'forward_ms': (t3 - t2) * 1000,
                    'post_ms': (t4 - t3) * 1000,
                    'model_ms': (t4 - t1) * 1000,
                    'total_ms': (t4 - t0) * 1000,
                }
                for key, value in times.items():
                    phases[key].append(value)

                target = int(target)
                rows.append({
                    'index': idx,
                    'target': target,
                    'target_name': class_names[target],
                    'prediction': prediction,
                    'prediction_name': class_names[prediction],
                    'correct': int(prediction == target),
                    'confidence': confidence,
                    **times,
                })

        if on_cuda:
            peak_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
        else:
            peak_mb = peak_rss_mb
        memory = {
            'kind': 'cuda_allocated' if on_cuda else 'cpu_rss',
            'baseline_mb': baseline_mb,
            'peak_mb': peak_mb,
            'overhead_mb': (peak_mb - baseline_mb) if baseline_mb is not None else None,
        }

        timing = {k: self._timing_stats(v) for k, v in phases.items()}
        latency = timing['total_ms']['mean_ms']
        summary = {
            'split': split_name,
            'num_samples': len(rows),
            'warmup_runs': warmup_runs,
            'seed': seed,
            'batch_size': 1,
            'device': str(self.device),
            'device_name': self._device_name(),
            'precision': self._precision,
            'img_size': self.img_size,
            'feedback_steps': self.feedback_steps,
            'latency_ms': latency,
            'model_latency_ms': timing['model_ms']['mean_ms'],
            'forward_latency_ms': timing['forward_ms']['mean_ms'],
            'throughput_imgs_per_sec': 1000.0 / latency if latency > 0 else None,
            'sample_accuracy': float(np.mean([r['correct'] for r in rows])),
            'timing': timing,
            'memory': memory,
            '_rows': rows,
        }
        summary['_report'] = self._emit(self._single_inference_lines(summary))
        return summary

# ======================================================================
# Stacking (RandomForest / XGBoost / CatBoost / LogReg meta-learner) inference
# ======================================================================

STACKING_HOWTO = """
The stacking artefacts are produced in three steps:

  1. python export_probs.py --data_path <RSCD> --resume output/<run>/best.pt \\
         --out_dir stacking/preds                     (JIT env)
         -> stacking/preds/jit_val.npz, jit_test.npz

  2. python export_probs_mamba.py ...                 (rscd_mamba env)
         -> mamba_val.npz, mamba_test.npz, copied into the same preds dir

  3. python stacking.py --preds_dir stacking/preds --meta_learner {rf,xgboost,catboost,logreg}
         -> stacking/<meta_learner>_meta.joblib
"""


class StackingInference(BaseEvaluator):
    """
    Evaluation of the stacking meta-learner (RandomForest, XGBoost, CatBoost or
    multinomial logistic regression)
    trained by `stacking.py`.  Everything that depends on the kind of
    meta-learner goes through `meta_learners`.

    Unlike the other two models this one is a scikit-learn estimator, not an
    `nn.Module`: it consumes the softmax probabilities that JITrans and
    RSCDMamba exported to `.npz`, aligned per image through `rel_paths`.  It
    therefore needs no GPU, no image transforms and no dataset - except for
    `plot_misclassified`, which needs `--data-path` to fetch the source images.

    Caveat that the report always states: the meta-learner is *trained* on the
    val split (stacking.py), so its val metrics are in-sample and
    optimistic.  Test is the honest number.
    """

    # Completed with the meta-learner in __init__ (e.g. 'stacking_xgboost')
    MODEL_NAME = 'stacking'
    # Names shown in the --compare plots and table; the keys stay the internal ones
    BASE_DISPLAY_NAMES = {'jit': 'JITClassifier', 'mamba': 'RoadRunnerMamba'}

    def __init__(
        self,
        meta_path: str,
        preds_dir: str,
        split: str = 'test',
        data_path: Optional[str] = None,
        data_set: str = 'RSCD',
        img_size: int = 224,
        eval_crop_ratio: float = 0.875,
        class_names: Optional[Sequence[str]] = None,
        base_files: Optional[Sequence[str]] = None,
        base_search_dirs: Optional[Sequence[str]] = None,
        **kwargs
    ):
        # `stacking.py` imports torch at module level (for the log-prob metric
        # trick), so reusing its helpers pulls torch in even though the
        # meta-learner itself never touches it.
        require_eval_deps()
        if split not in ('val', 'test', 'both', 'separate'):
            raise ValueError(
                f"split must be 'val', 'test', 'both' or 'separate', got '{split}'"
            )

        import joblib
        from stacking import load_preds, align_split
        from meta_learners import display_name, meta_learner_name

        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Stacking meta-model not found: {meta_path}\n{STACKING_HOWTO}")

        bundle = joblib.load(meta_path)
        # Bundles written before the meta-learner choice hold the RandomForest under 'rf'
        self.meta = bundle['model'] if 'model' in bundle else bundle['rf']
        self.meta_learner = bundle.get('meta_learner') or meta_learner_name(self.meta)
        self.meta_display = display_name(self.meta_learner)
        self.MODEL_NAME = f"stacking_{self.meta_learner}"
        meta_classes = [str(c) for c in bundle['classes']]
        self.feature_order = bundle.get('feature_order', ['jit', 'mamba'])
        # Bundles written before --features was added were trained on the probabilities.
        # The blocks below keep the probabilities (the base-model metrics slice them);
        # the transform is applied only on the way into the meta-learner.
        self.features = bundle.get('features', 'probs')
        print(f"{OK} Stacking meta-model loaded: {meta_path}")
        print(f"  meta-learner:  {self.meta_display} ({type(self.meta).__name__})")
        print(f"  feature order: {self.feature_order} ({self.features})")

        # Load and align the exported probabilities for the needed splits.
        needed = ['val', 'test'] if split in ('both', 'separate') else [split]
        self._blocks: Dict[str, Dict[str, Any]] = {}
        for name in needed:
            paths = {model: os.path.join(preds_dir, f"{model}_{name}.npz")
                     for model in self.feature_order}
            missing = [p for p in paths.values() if not os.path.isfile(p)]
            if missing:
                raise FileNotFoundError(
                    "Missing exported probability file(s):\n  "
                    + "\n  ".join(missing) + "\n" + STACKING_HOWTO)

            preds = {model: load_preds(path) for model, path in paths.items()}
            X, y, classes, rel_paths = align_split(
                preds[self.feature_order[0]], preds[self.feature_order[1]],
                name, return_rel_paths=True)
            self._blocks[name] = {'X': X, 'y': y, 'rel_paths': rel_paths,
                                  'classes': classes}

        # Every block must agree on the class ordering, and so must the meta-learner.
        all_classes = {tuple(b['classes']) for b in self._blocks.values()}
        if len(all_classes) > 1:
            raise ValueError(f"Class ordering differs between splits: {all_classes}")
        classes = list(next(iter(all_classes)))
        if meta_classes != classes:
            print(f"{WARN} class list in the joblib {meta_classes} differs from the "
                  f"npz files {classes}; using the npz ordering")

        if split == 'both':
            self._blocks = {'both': {
                'X': np.concatenate([self._blocks[n]['X'] for n in needed], axis=0),
                'y': np.concatenate([self._blocks[n]['y'] for n in needed], axis=0),
                'rel_paths': [p for n in needed for p in self._blocks[n]['rel_paths']],
                'classes': classes,
            }}

        super().__init__(
            num_classes=len(classes),
            split_names=list(self._blocks.keys()),
            class_names=list(class_names) if class_names else classes,
            model_path=meta_path,
            split=split,
        )

        # Optional image access for plot_misclassified
        self.data_path = data_path
        self.data_set = data_set
        self.img_size = img_size
        self.eval_crop_ratio = eval_crop_ratio
        self._image_datasets: Dict[str, Any] = {}
        self._rel_to_index: Dict[str, Dict[str, int]] = {}

        # JSON reports of the base models (model_summary_*.json and
        # single_inference_*.json), used for the full-pipeline figures.
        self.base_files = list(base_files or [])
        for path in self.base_files:
            if not os.path.isfile(path):
                print(f"{WARN} --base-files: {path} not found, ignored")
        seen = set()
        self.base_search_dirs = []
        for directory in list(base_search_dirs or []) + [preds_dir,
                                                        os.path.dirname(os.path.abspath(meta_path))]:
            if directory and os.path.abspath(directory) not in seen:
                seen.add(os.path.abspath(directory))
                self.base_search_dirs.append(directory)

        print(f"{OK} Stacking inference setup complete")
        print(f"  Split mode:   {self.split.upper()}")
        for name in self.split_names:
            print(f"    - {name:<10} {len(self._blocks[name]['y'])} aligned samples "
                  f"({self._blocks[name]['X'].shape[1]} meta-features)")
        print(f"  Num classes:  {self.num_classes}")
        print(f"  Class names:  {', '.join(self._class_names())}")
        if not data_path:
            print(f"  {WARN} no --data-path: the misclassified-images plot will be skipped")

    # -- hooks --------------------------------------------------------

    def _split_note(self, split_name: str) -> Optional[str]:
        if split_name in ('val', 'both'):
            return (f"the {self.meta_display} meta-learner is TRAINED on the val split "
                    "(stacking.py), so these numbers are in-sample and "
                    "optimistic; use the test split for an honest estimate")
        return None

    def _predict_split(
        self, split_name: str, verbose: bool = True, track_performance: bool = True
    ):
        block = self._blocks[split_name]
        X, y = block['X'], block['y']

        if verbose:
            print(f"Running the meta-learner on {len(y)} aligned samples "
                  f"(split={split_name.upper()})...")

        start_time = time.perf_counter()
        proba_raw = self._raw_predict_proba(X)
        elapsed = time.perf_counter() - start_time

        # classes_ is not guaranteed to be 0..C-1 in order, so reorder the
        # columns into the canonical class order (same as stacking.py).
        probabilities = self._reorder_proba(proba_raw)

        performance = None
        if track_performance:
            performance = {
                'device': f'CPU ({self.meta_display})',
                'meta_learner': self.meta_learner,
                'total_time_s': elapsed,
                'total_images': len(y),
                'batch_size': None,
                'num_batches': None,
                'mean_batch_time_ms': None,
                'std_batch_time_ms': None,
                'median_batch_time_ms': None,
                'mean_time_per_image_ms': elapsed * 1000.0 / max(len(y), 1),
                'throughput_imgs_per_sec': len(y) / elapsed if elapsed > 0 else float('inf'),
                'n_estimators': self.meta.get_params().get(
                    'n_estimators', self.meta.get_params().get('iterations')),
                'n_meta_features': int(X.shape[1]),
                # Note: this is the meta-learner only; it excludes the two base
                # models' forward passes, which dominate end-to-end latency.
                'note': 'meta-learner only, excludes the base models forward passes',
            }

        indices = np.arange(len(y))
        return probabilities, y, indices, performance

    def _evaluate_one(self, split_name: str, **kwargs) -> Dict:
        results = super()._evaluate_one(split_name, **kwargs)
        # Carried through so `--save-predictions` can name each row.
        results['_rel_paths'] = self._blocks[split_name]['rel_paths']
        return results

    def _get_image(self, split_name: str, index: int) -> torch.Tensor:
        """
        Map an aligned row back to its image on disk.

        The row order comes from `align_split`, which sorts the rel_paths, so it
        does NOT match the ImageFolder order: the lookup goes through the
        rel_path key, exactly like the alignment itself.
        """
        if not self.data_path:
            raise NotImplementedError(
                "pass --data-path to fetch the source images for the stacking model")
        if split_name == 'both':
            raise NotImplementedError(
                "image lookup is ambiguous with --split both; use 'separate'")

        if split_name not in self._image_datasets:
            self._build_image_dataset(split_name)

        rel_path = self._blocks[split_name]['rel_paths'][index]
        dataset_index = self._rel_to_index[split_name].get(rel_path)
        if dataset_index is None:
            raise KeyError(f"{rel_path} not found in the {split_name} ImageFolder")
        return self._image_datasets[split_name][dataset_index][0]

    def _build_image_dataset(self, split_name: str):
        """Lazily build the ImageFolder and the rel_path -> index lookup."""
        from dataloader import build_dataset
        from export_probs import rel_paths_from_dataset

        class Args:
            pass

        args = Args()
        args.data_path = self.data_path
        args.data_set = self.data_set
        args.img_size = self.img_size
        args.eval_crop_ratio = self.eval_crop_ratio
        args.color_jitter = 0.4
        args.aa = 'rand-m9-mstd0.5-inc1'
        args.train_interpolation = 'bicubic'
        args.reprob = 0.25
        args.remode = 'pixel'
        args.recount = 1

        dataset, _ = build_dataset(
            is_train=False, is_test=(split_name == 'test'), args=args)
        self._image_datasets[split_name] = dataset
        self._rel_to_index[split_name] = {
            rel: i for i, rel in enumerate(rel_paths_from_dataset(dataset))
        }
        print(f"  {OK} {split_name} ImageFolder loaded for image lookup "
              f"({len(dataset)} samples)")

    # ------------------------------------------------------------------
    # Complexity / latency
    # ------------------------------------------------------------------
    #
    # The torch-based `calculate_model_complexity` of JITrans (parameters +
    # FlopCounterMode on a forward pass) does not apply to a scikit-learn /
    # boosting estimator: it has no tensors (a tree ensemble executes no
    # floating-point layers at all; the logistic regression FLOPs are counted
    # analytically from its coefficients).  The
    # equivalents below describe the meta-learner alone; the base models,
    # which dominate the cost of the ensemble, run in their own conda envs
    # and are folded in from the JSON reports they export.

    @contextlib.contextmanager
    def _single_job(self):
        """
        Run the meta-learner on one thread for per-sample timing.

        The meta-learner keeps the threads it was trained with (stacking.py
        --n_jobs); with -1 every `predict_proba` call on one row pays the
        thread pool start-up, which dwarfs the actual tree traversal (the
        logistic regression has its BLAS pool limited instead).
        Inside this context `_raw_predict_proba` runs single-threaded.
        """
        from meta_learners import single_thread

        with single_thread(self.meta) as original:
            self._predict_single_thread = True
            try:
                yield original
            finally:
                self._predict_single_thread = False

    def _raw_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """`predict_proba` with the columns in `classes_` order.

        X holds the stacked probabilities; the --features transform of stacking.py is
        applied here, so it is part of the timed meta-learner call.
        """
        from meta_learners import meta_features, raw_predict_proba

        return raw_predict_proba(self.meta, meta_features(X, self.features),
                                 getattr(self, '_predict_single_thread', False))

    def _features_label(self) -> str:
        kind = 'log-probabilities' if self.features == 'logprobs' else 'probabilities'
        return f"{' + '.join(self.feature_order)} {kind}"

    def _default_block(self, split_name: Optional[str] = None) -> Tuple[str, Dict]:
        """The split used for the analyses: test when available (val is in-sample)."""
        split_name = split_name or ('test' if 'test' in self._blocks
                                    else self.split_names[0])
        return split_name, self._blocks[split_name]

    def _reorder_proba(self, proba_raw: np.ndarray) -> np.ndarray:
        """Map the `classes_` columns onto the canonical class order."""
        from meta_learners import reorder_proba

        return reorder_proba(self.meta, proba_raw, self.num_classes)

    @staticmethod
    def _read_json(path: str) -> Optional[Dict]:
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except (OSError, ValueError) as exc:
            print(f"{WARN} could not read {path}: {exc}")
            return None

    def _load_base_files(self, kind: str) -> Dict[str, Dict]:
        """
        Find the base-model reports of one kind ('model_summary' or
        'single_inference'): first the explicit --base-files, then
        `<kind>_<model>.json` in the search directories.
        """
        found: Dict[str, Dict] = {}
        for path in self.base_files:
            if not os.path.isfile(path):
                continue
            data = self._read_json(path)
            if data and data.get('kind') == kind and data.get('model') in self.feature_order:
                found.setdefault(data['model'], dict(data, _path=path))
        for directory in self.base_search_dirs:
            for model in self.feature_order:
                path = os.path.join(directory, f"{kind}_{model}.json")
                if model not in found and os.path.isfile(path):
                    data = self._read_json(path)
                    if data:
                        found[model] = dict(data, _path=path)
        return found

    def calculate_model_complexity(self, verbose: bool = True,
                                   max_path_samples: int = 2000) -> Dict:
        """
        Complexity of the meta-learner.

        Tree ensembles (RF / XGBoost / CatBoost):
        Size:  trees, nodes, leaves, depth, stored values (feature index +
               threshold per split, leaf values), serialized size.
        Cost:  split comparisons per sample, measured on the real
               meta-features (see `meta_learners.tree_ensemble_stats` for the
               per-library method), plus the leaf-value additions.  This is
               the tree-ensemble analogue of FLOPs; it is NOT in the same unit.

        Logistic regression: see `_linear_model_complexity`.
        """
        import pickle
        from meta_learners import is_tree_ensemble, meta_features, n_threads, tree_ensemble_stats

        meta = self.meta
        if not is_tree_ensemble(meta):
            return self._linear_model_complexity(verbose)
        n_classes = len(meta.classes_)

        split_name, block = self._default_block()
        X = block['X']
        if len(X) > max_path_samples:
            X = X[np.sort(self._sample_indices(len(X), max_path_samples, 0))]
        stats = tree_ensemble_stats(meta, meta_features(X, self.features))
        comparisons = stats['comparisons_per_sample']
        accumulation = stats['additions_per_sample']
        stored_values = stats['stored_values']
        serialized_mb = len(pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)) / 1024**2
        file_mb = (os.path.getsize(self.model_path) / 1024**2
                   if self.model_path and os.path.isfile(self.model_path) else None)
        params = meta.get_params()
        n_estimators = params.get('n_estimators', params.get('iterations'))

        results = {
            'estimator': type(meta).__name__,
            'meta_learner': self.meta_learner,
            'n_estimators': int(n_estimators) if n_estimators is not None else None,
            'n_trees': stats['n_trees'],
            'n_features_in': int(getattr(meta, 'n_features_in_', X.shape[1])),
            'features': self.features,
            'n_classes': n_classes,
            'max_depth_param': stats['max_depth_param'],
            'total_nodes': stats['total_nodes'],
            'total_split_nodes': stats['total_split_nodes'],
            'total_leaves': stats['total_leaves'],
            'mean_nodes_per_tree': stats['mean_nodes_per_tree'],
            'max_depth': stats['max_depth'],
            'mean_depth': stats['mean_depth'],
            'stored_values': stored_values,
            'stored_values_formatted': self._format_count(stored_values),
            'trainable_params': 0,
            'model_size_mb': serialized_mb,
            'checkpoint_size_mb': file_mb,
            'nodes_visited_per_sample': float(comparisons + stats['n_trees']),
            'comparisons_per_sample': float(comparisons),
            'ops_per_sample': float(comparisons + accumulation),
            'ops_convention': stats['cost_convention'],
            'ops_measured_on': f"{split_name} ({X.shape[0]} rows)",
            'flops': None,
            'flops_note': f'not applicable: a {self.meta_display} ensemble has no '
                          f'floating-point layers',
            'n_jobs_trained': n_threads(meta),
            'method': f'{self.meta_learner}_tree_stats',
        }

        size_file = f"{file_mb:.2f} MB" if file_mb is not None else 'N/A'
        lines = [
            "=" * 60,
            f"META-LEARNER COMPLEXITY ({self.meta_display})",
            "=" * 60,
            f"Estimator:             {results['estimator']}  ({stats['n_trees']} trees)",
            f"Meta-features:         {results['n_features_in']}  "
            f"({self._features_label()})",
            f"Nodes / leaves:        {results['total_nodes']:,} / {results['total_leaves']:,}",
            f"Depth (max / mean):    {results['max_depth']} / {results['mean_depth']:.1f}",
            f"Stored values:         {results['stored_values_formatted']} ({stored_values:,})"
            f"  [trainable torch params: 0]",
            f"Size (serialized):     {serialized_mb:.2f} MB",
            f"joblib on disk:        {size_file}",
            f"Comparisons / sample:  {comparisons:,.1f}",
            f"Operations / sample:   {results['ops_per_sample']:,.1f}  "
            f"(comparisons + {accumulation:,.0f} additions)",
            f"Measured on:           {results['ops_measured_on']}",
            "FLOPs:                 N/A (no floating-point layers)",
            "=" * 60,
        ]
        results['_report'] = self._emit(lines) if verbose else lines
        return results

    def _linear_model_complexity(self, verbose: bool = True) -> Dict:
        """
        Complexity of the logistic-regression meta-learner.

        Size:  coefficients + intercepts (the trainable parameters, which live
               in NumPy, not torch), serialized size.
        Cost:  genuine FLOPs, computed analytically from the coefficient matrix
               (see `meta_learners.linear_model_stats`); they do not depend on
               the input, so no sample is needed.
        The tree-only keys are kept as None so the result has a single schema.
        """
        import pickle
        from meta_learners import linear_model_stats

        meta = self.meta
        stats = linear_model_stats(meta)
        total_params = stats['total_params']
        flops = stats['flops_per_sample']
        macs = stats['macs_per_sample']
        serialized_mb = len(pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)) / 1024**2
        file_mb = (os.path.getsize(self.model_path) / 1024**2
                   if self.model_path and os.path.isfile(self.model_path) else None)
        fmt = self._format_count

        results = {
            'estimator': type(meta).__name__,
            'meta_learner': self.meta_learner,
            'n_estimators': None,
            'n_trees': None,
            'n_features_in': stats['n_features'],
            'features': self.features,
            'n_classes': stats['n_classes_fitted'],
            'max_depth_param': None,
            'total_nodes': None,
            'total_split_nodes': None,
            'total_leaves': None,
            'mean_nodes_per_tree': None,
            'max_depth': None,
            'mean_depth': None,
            'n_weights': stats['n_weights'],
            'n_biases': stats['n_biases'],
            'stored_values': total_params,
            'stored_values_formatted': fmt(total_params),
            'trainable_params': total_params,
            'model_size_mb': serialized_mb,
            'checkpoint_size_mb': file_mb,
            'nodes_visited_per_sample': None,
            'comparisons_per_sample': None,
            'ops_per_sample': float(flops),
            'ops_convention': stats['cost_convention'],
            'ops_measured_on': 'analytic (coefficient shape)',
            'flops': flops,
            'flops_formatted': fmt(flops),
            'macs': macs,
            'macs_formatted': fmt(macs),
            'flops_convention': stats['cost_convention'],
            'C': meta.C,
            'n_iter': int(np.max(meta.n_iter_)),
            'max_iter': meta.max_iter,
            'n_jobs_trained': None,
            'method': f'{self.meta_learner}_analytic',
        }

        size_file = f"{file_mb:.4f} MB" if file_mb is not None else 'N/A'
        lines = [
            "=" * 60,
            f"META-LEARNER COMPLEXITY ({self.meta_display})",
            "=" * 60,
            f"Estimator:             {results['estimator']}  (C={meta.C:g}, "
            f"{results['n_iter']}/{meta.max_iter} lbfgs iterations)",
            f"Meta-features:         {results['n_features_in']}  "
            f"({self._features_label()})",
            f"Parameters:            {fmt(total_params)} ({total_params:,})  "
            f"[{stats['n_weights']:,} weights + {stats['n_biases']:,} biases, NumPy]",
            f"Size (serialized):     {serialized_mb:.4f} MB",
            f"joblib on disk:        {size_file}",
            f"MACs / sample:         {macs:,}",
            f"FLOPs / sample:        {flops:,}",
            f"Convention:            {stats['cost_convention']}",
            "=" * 60,
        ]
        results['_report'] = self._emit(lines) if verbose else lines
        return results

    def pipeline_complexity(self, meta: Dict) -> Optional[Dict]:
        """
        Complexity of the whole ensemble: JITrans + Mamba + meta-learner.

        The base-model numbers are read from their model_summary_<model>.json
        (written by `--model-summary` in each repo).  FLOPs are summed only
        when both reports use the same convention; the meta-learner operations
        (tree ensembles: a different unit; logistic regression: FLOPs counted
        analytically, negligible next to the base models) stay separate.
        """
        bases = self._load_base_files('model_summary')
        if not bases:
            lines = [f"{WARN} no model_summary_<jit|mamba>.json found "
                     f"(searched {self.base_search_dirs} and --base-files): "
                     f"full-pipeline complexity skipped"]
            self._emit(lines)
            return None

        per_model: Dict[str, Dict] = {}
        warnings: List[str] = []
        for model in self.feature_order:
            data = bases.get(model)
            if data is None:
                continue
            c = data.get('complexity') or {}
            per_model[model] = {
                'total_params': c.get('total_params'),
                'flops': c.get('flops'),
                'macs': c.get('macs'),
                'model_size_mb': c.get('model_size_mb'),
                'flops_convention': c.get('flops_convention'),
                'method': c.get('method'),
                'source': data['_path'],
            }
            if c.get('flops_warning'):
                warnings.append(f"{model}: {c['flops_warning']}")

        missing = [m for m in self.feature_order if m not in per_model]
        if missing:
            warnings.append(f"missing model summary for {missing}: totals are partial")
        conventions = {v['flops_convention'] for v in per_model.values()}
        same_convention = len(conventions) == 1

        def _sum(key):
            values = [v[key] for v in per_model.values()]
            if not values or any(v is None for v in values):
                return None
            return sum(values)

        flops = _sum('flops') if same_convention else None
        macs = _sum('macs') if same_convention else None
        if not same_convention:
            warnings.append(f"FLOPs conventions differ {sorted(map(str, conventions))}: "
                            f"not summed")
        base_size = _sum('model_size_mb')

        result = {
            'base_models': per_model,
            'complete': not missing,
            'total_params': _sum('total_params'),
            'flops': flops,
            'macs': macs,
            'gflops': flops / 1e9 if flops is not None else None,
            'gmacs': macs / 1e9 if macs is not None else None,
            'meta_learner_ops_per_sample': meta.get('ops_per_sample'),
            'meta_learner_flops': meta.get('flops'),
            'meta_learner_stored_values': meta.get('stored_values'),
            'model_size_mb': (base_size + meta['model_size_mb']
                              if base_size is not None else None),
            'warnings': warnings,
        }

        fmt = self._format_count
        lines = ["=" * 60, "FULL-PIPELINE COMPLEXITY (base models + meta-learner)", "=" * 60]
        for model, v in per_model.items():
            lines.append(f"{model:<8} params {fmt(v['total_params']):>9}   "
                         f"FLOPs {fmt(v['flops']):>9}   MACs {fmt(v['macs']):>9}   "
                         f"[{v['method']}]")
        if meta.get('flops') is not None:
            lines.append(f"{self.meta_learner:<8} params {meta['stored_values_formatted']:>9}   "
                         f"FLOPs {fmt(meta['flops']):>9}   MACs {fmt(meta['macs']):>9}   "
                         f"[{meta['method']}, not summed]")
        else:
            lines.append(f"{self.meta_learner:<8} stored values {meta['stored_values_formatted']}, "
                         f"{meta['ops_per_sample']:,.0f} ops/sample (not FLOPs)")
        lines += [
            "-" * 60,
            f"Total params (base):   {fmt(result['total_params'])}",
            f"Total FLOPs (base):    {fmt(flops)}"
            + (f"  ({result['gflops']:.3f} GFLOPs)" if flops is not None else ''),
            f"Total MACs (base):     {fmt(macs)}"
            + (f"  ({result['gmacs']:.3f} GMACs)" if macs is not None else ''),
        ]
        if result['model_size_mb'] is not None:
            lines.append(f"Total size (memory):   {result['model_size_mb']:.2f} MB")
        lines += [f"{WARN} {w}" for w in warnings]
        lines.append("=" * 60)
        result['_report'] = self._emit(lines)
        return result

    def benchmark_inference(self, num_runs: int = 100, warmup_runs: int = 10) -> Dict:
        """`predict_proba` latency on one row (batch = 1), with n_jobs = 1."""
        _, block = self._default_block()
        X = block['X']
        rng = np.random.default_rng(0)
        rows = rng.integers(0, len(X), size=num_runs + warmup_runs)

        print(f"Running {num_runs} single-row predict_proba calls "
              f"(+{warmup_runs} warm-up)...")
        times = []
        with self._single_job() as n_jobs_trained:
            for i, row in enumerate(rows):
                sample = X[row:row + 1]
                start = time.perf_counter()
                self._raw_predict_proba(sample)
                if i >= warmup_runs:
                    times.append((time.perf_counter() - start) * 1000)

        times = np.array(times)
        results = {
            'mean_time_ms': float(np.mean(times)),
            'std_time_ms': float(np.std(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times)),
            'median_time_ms': float(np.median(times)),
            'p95_time_ms': float(np.percentile(times, 95)),
            'p99_time_ms': float(np.percentile(times, 99)),
            'fps': 1000.0 / float(np.mean(times)),
            'throughput': 1000.0 / float(np.mean(times)),
            'input_size': [1, int(X.shape[1])],
            'device': f'cpu ({self.meta_display})',
            'meta_learner': self.meta_learner,
            'device_name': self._device_name(),
            'n_jobs': 1,
            'n_jobs_trained': n_jobs_trained,
            'num_runs': num_runs,
            'warmup_runs': warmup_runs,
        }
        lines = [
            "=" * 60,
            f"META-LEARNER BENCHMARK ({self.meta_display} predict_proba, batch = 1)",
            "=" * 60,
            f"Mean Time:    {results['mean_time_ms']:.3f} +/- {results['std_time_ms']:.3f} ms",
            f"Median Time:  {results['median_time_ms']:.3f} ms",
            f"P95 / P99:    {results['p95_time_ms']:.3f} / {results['p99_time_ms']:.3f} ms",
            f"Min/Max Time: {results['min_time_ms']:.3f} / {results['max_time_ms']:.3f} ms",
            f"Throughput:   {results['throughput']:.2f} samples/sec",
            f"Device:       {results['device']} ({results['device_name']})",
            f"n_jobs:       1 for the timing (trained with {n_jobs_trained})"
            if n_jobs_trained is not None else
            "Threads:      BLAS limited to 1 for the timing (threadpoolctl)",
            "=" * 60,
        ]
        print()
        results['_report'] = self._emit(lines)
        return results

    def model_summary(
        self,
        include_benchmark: bool = False,
        include_memory: bool = False,
        benchmark_runs: int = 100,
        **kwargs
    ) -> Dict:
        """Meta-learner complexity, full-pipeline complexity and latency benchmark."""
        print("Starting Model Analysis...")
        print("=" * 60)

        complexity = self.calculate_model_complexity(verbose=True)
        print()
        results = {
            'complexity': complexity,
            'pipeline': self.pipeline_complexity(complexity),
            'benchmark': None,
            'memory': None,
        }

        if include_benchmark:
            print("\n" + "=" * 60)
            results['benchmark'] = self.benchmark_inference(benchmark_runs, warmup_runs=10)

        if include_memory:
            lines = [f"{WARN} memory profiling does not apply to the {self.meta_display} "
                     f"(no device tensors): see the serialized size above "
                     f"({complexity['model_size_mb']:.2f} MB)"]
            results['memory'] = {'note': lines[0], '_report': self._emit(lines)}

        print("\n" + "=" * 60)
        print(f"{OK} Model Analysis Complete!")
        print("=" * 60)
        return results

    def single_image_inference(
        self,
        num_samples: int = 50,
        warmup_runs: int = 10,
        seed: Optional[int] = 42,
        split_name: Optional[str] = None,
    ) -> Dict:
        """
        Latency of one stacking prediction at a time (batch = 1).

        Measured here: the meta-learner (`predict_proba` on one meta-feature
        row, n_jobs = 1) plus the column reorder / argmax.  The base models
        cannot run in this env, so the ensemble latency is composed from
        their single_inference_<model>.json (written by `--inference` in
        each repo):
            sequential = jit + mamba + meta        (one device, models in series)
            parallel   = max(jit, mamba) + meta    (two devices / streams)
        both for the end-to-end latency (each model loads and preprocesses
        the image itself) and for the model-only latency (image already
        preprocessed).
        """
        split_name, block = self._default_block(split_name)
        X, y = block['X'], block['y']
        class_names = self._class_names(split_name)
        indices = self._sample_indices(len(y), num_samples, seed)

        print(f"\nSingle-image inference: {len(indices)} meta-feature rows from "
              f"{split_name.upper()} (batch=1, warm-up {warmup_runs} runs, n_jobs=1)")

        phases: Dict[str, List[float]] = {'predict_ms': [], 'post_ms': [], 'total_ms': []}
        rows: List[Dict[str, Any]] = []
        with self._single_job() as n_jobs_trained:
            first = int(indices[0])
            for _ in range(warmup_runs):
                self._raw_predict_proba(X[first:first + 1])

            for idx in indices:
                idx = int(idx)
                sample = X[idx:idx + 1]
                t0 = time.perf_counter()
                proba_raw = self._raw_predict_proba(sample)
                t1 = time.perf_counter()
                probs = self._reorder_proba(proba_raw)[0]
                prediction = int(probs.argmax())
                confidence = float(probs[prediction])
                t2 = time.perf_counter()

                times = {'predict_ms': (t1 - t0) * 1000,
                         'post_ms': (t2 - t1) * 1000,
                         'total_ms': (t2 - t0) * 1000}
                for key, value in times.items():
                    phases[key].append(value)
                target = int(y[idx])
                rows.append({
                    'index': idx,
                    'rel_path': block['rel_paths'][idx],
                    'target': target,
                    'target_name': class_names[target],
                    'prediction': prediction,
                    'prediction_name': class_names[prediction],
                    'correct': int(prediction == target),
                    'confidence': confidence,
                    **times,
                })

        timing = {k: self._timing_stats(v) for k, v in phases.items()}
        meta_latency = timing['total_ms']['mean_ms']
        pipeline, pipeline_lines = self._pipeline_latency(meta_latency)

        summary = {
            'split': split_name,
            'num_samples': len(rows),
            'warmup_runs': warmup_runs,
            'seed': seed,
            'batch_size': 1,
            'device': f'cpu ({self.meta_display})',
            'device_name': self._device_name(),
            'precision': f'{self.meta_display} predict_proba',
            'meta_learner': self.meta_learner,
            'n_jobs': 1,
            'n_jobs_trained': n_jobs_trained,
            'latency_ms': meta_latency,
            'meta_learner_latency_ms': meta_latency,
            'pipeline_latency_ms': pipeline.get('sequential_ms') if pipeline else None,
            'throughput_imgs_per_sec': 1000.0 / meta_latency if meta_latency > 0 else None,
            'sample_accuracy': float(np.mean([r['correct'] for r in rows])),
            'timing': timing,
            'pipeline': pipeline,
            '_rows': rows,
        }
        summary['_report'] = self._emit(
            self._single_inference_lines(summary, extra=pipeline_lines))
        return summary

    def _pipeline_latency(self, meta_ms: float) -> Tuple[Optional[Dict], List[str]]:
        """Compose the ensemble latency from the base models' timing reports."""
        bases = self._load_base_files('single_inference')
        missing = [m for m in self.feature_order if m not in bases]
        if missing:
            lines = [f"{WARN} single_inference_<model>.json missing for {missing} "
                     f"(searched {self.base_search_dirs} and --base-files):",
                     "    only the meta-learner latency is reported. Run --inference "
                     "in each base-model repo first."]
            return None, lines

        per_model = {}
        for model in self.feature_order:
            data = bases[model]
            per_model[model] = {
                'latency_ms': data.get('latency_ms'),
                'model_latency_ms': data.get('model_latency_ms'),
                'forward_latency_ms': data.get('forward_latency_ms'),
                'device': data.get('device'),
                'device_name': data.get('device_name'),
                'precision': data.get('precision'),
                'num_samples': data.get('num_samples'),
                'source': data['_path'],
            }

        warnings = []
        if len({v['device_name'] for v in per_model.values()}) > 1:
            warnings.append("the base models were timed on different devices "
                            f"({', '.join(str(v['device_name']) for v in per_model.values())}): "
                            "the sum mixes hardware")
        if len({v['precision'] for v in per_model.values()}) > 1:
            warnings.append("the base models were timed with different precisions")

        def _compose(key):
            values = [v[key] for v in per_model.values()]
            if any(v is None for v in values):
                return None, None
            return sum(values) + meta_ms, max(values) + meta_ms

        seq, par = _compose('latency_ms')
        seq_model, par_model = _compose('model_latency_ms')
        pipeline = {
            'base_models': per_model,
            'meta_learner': self.meta_learner,
            'meta_learner_ms': meta_ms,
            'sequential_ms': seq,
            'parallel_ms': par,
            'sequential_model_only_ms': seq_model,
            'parallel_model_only_ms': par_model,
            'sequential_throughput_imgs_per_sec': 1000.0 / seq if seq else None,
            'warnings': warnings,
        }

        def _ms(value):
            return f"{value:9.3f}" if value is not None else f"{'N/A':>9}"

        lines = ["ENSEMBLE LATENCY PER IMAGE (composed)",
                 f"{'':<22}{'end-to-end':>12}{'model-only':>12}   [ms]"]
        for model, v in per_model.items():
            lines.append(f"{model:<22}"
                         f"{_ms(v['latency_ms']):>12}{_ms(v['model_latency_ms']):>12}")
        lines += [
            f"{self.meta_learner + ' (meta-learner)':<22}{_ms(meta_ms):>12}{_ms(meta_ms):>12}",
            "-" * 60,
            f"{'sequential':<22}{_ms(seq):>12}{_ms(seq_model):>12}",
            f"{'parallel (lower bound)':<22}{_ms(par):>12}{_ms(par_model):>12}",
        ]
        if seq:
            lines.append(f"Sequential throughput:  {1000.0 / seq:.2f} images/sec")
        for model, v in per_model.items():
            lines.append(f"{model} timed on {v['device_name']} ({v['precision']}, "
                         f"{v['num_samples']} images) <- {v['source']}")
        lines += [f"{WARN} {w}" for w in warnings]
        return pipeline, lines

    # ------------------------------------------------------------------
    # Three-way model comparison
    # ------------------------------------------------------------------

    def compare_models(
        self,
        split_name: str,
        verbose: bool = True,
        out_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
        show: bool = False,
        dpi: int = 150,
        plot_format: str = 'pdf',
    ) -> Dict[str, Dict]:
        """
        Evaluate the two base models and the stacking meta-learner on the very
        same aligned rows, and print / export the comparison table.

        The base-model probabilities come straight out of the aligned feature
        matrix, so no forward pass and no checkpoint is needed here.
        """
        from stacking import base_model_probs_from_X

        block = self._blocks[split_name]
        X, y = block['X'], block['y']
        class_names = self._class_names(split_name)

        results_by_model: Dict[str, Dict] = {}

        for model_name in self.feature_order:
            probs = base_model_probs_from_X(X, self.num_classes, model_name)
            results_by_model[model_name] = self._results_from_probs(
                probs, y, split_name, model_name, class_names)

        stacked_probs, _, _, _ = self._predict_split(
            split_name, verbose=False, track_performance=False)
        results_by_model[self.MODEL_NAME] = self._results_from_probs(
            stacked_probs, y, split_name, self.MODEL_NAME, class_names,
            note=self._split_note(split_name))

        if verbose:
            self.print_split_comparison(
                results_by_model,
                out_dir=out_dir,
                title=f"MODEL COMPARISON  [split={split_name.upper()}]",
                csv_name=f"comparison_{'_vs_'.join(results_by_model)}_{split_name}.csv",
            )

            best_base = max(results_by_model[m]['accuracy'] for m in self.feature_order)
            delta = results_by_model[self.MODEL_NAME]['accuracy'] - best_base
            print(f"Stacking vs best base model: {delta:+.4f} accuracy")

        if save_dir:
            for kind in ('roc', 'pr'):
                self._plot_curve_overlay(results_by_model, split_name, kind,
                                         save_dir, show, dpi, plot_format)

        return results_by_model

    def _results_from_probs(
        self, probs, targets, split_name, model_name, class_names, note=None
    ) -> Dict:
        """Wrap a probability matrix into the standard results dict."""
        probs = np.asarray(probs)
        predictions = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        results = {
            'split': split_name,
            'model': model_name,
            'n_samples': len(targets),
            'num_classes': self.num_classes,
            'class_names': class_names,
            '_probabilities': probs,
            '_targets': np.asarray(targets),
            '_predictions': predictions,
            '_confidences': confidences,
            '_indices': np.arange(len(targets)),
        }
        results['display_name'] = self._display_name(model_name)
        results.update(self._extended_metrics(
            np.asarray(targets), predictions, probs, confidences, class_names))
        if note:
            results['note'] = note
        return results

    def _display_name(self, model_name: str) -> str:
        """Readable name of a --compare model: e.g. 'stacking_rf' -> 'Stacking (RandomForest)'."""
        if model_name == self.MODEL_NAME:
            return f"Stacking ({self.meta_display})"
        return self.BASE_DISPLAY_NAMES.get(model_name, model_name)

    def _plot_curve_overlay(
        self, results_by_model, split_name, kind, save_dir, show, dpi, plot_format
    ) -> List[str]:
        """
        Macro-average ROC (kind='roc') or Precision-Recall (kind='pr') curve
        of the three models on one axes.  Macro rather than micro: the classes
        are imbalanced and micro would mostly reflect the largest ones; the
        micro score is still reported in the legend.
        """
        fig, ax = plt.subplots(figsize=(8, 7))
        colors = {'jit': '#1f77b4', 'mamba': '#ff7f0e', 'stacking': '#2ca02c'}
        metric = 'AUC' if kind == 'roc' else 'AP'
        lines = []

        if kind == 'pr':
            _draw_iso_f1(ax)

        for model_name, results in results_by_model.items():
            curves = results.get('_curves') or compute_curve_data(
                results['_targets'], results['_probabilities'], self.num_classes)
            if curves is None:
                continue
            scores = auc_scores_from_curves(curves, results['class_names'][:self.num_classes])
            if curves['macro'] is None:
                # Two classes: the positive-class curve
                x, y, _ = curves[kind][1 if 1 in curves[kind] else curves['valid'][0]]
            else:
                x, y, _ = curves['macro'][kind]
            style = dict(color=colors.get(model_name.split('_')[0]), lw=2)
            label = results.get('display_name', model_name)
            ax.plot(x, y, label=(f"{label}  (macro {metric} = {scores[f'{kind}_auc_macro']:.4f}, "
                                 f"micro = {scores[f'{kind}_auc_micro']:.4f})"), **style)
            lines.append((x, y, style))

        if kind == 'roc':
            ax.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.5, label="Chance level  (AUC = 0.5)")
            if lines and min(auc(x, y) for x, y, _ in lines) >= 0.95:
                x_max = 0.05
                y_min = min(_value_at(0.1 * x_max, x, y) for x, y, _ in lines)
                _add_zoom_inset(ax, lines, (0, x_max), (y_min, 1.0), [0.35, 0.35, 0.4, 0.4])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            title, legend_loc = "Macro-average ROC", 'lower right'
        else:
            if lines and min(_value_at(0.9, x, y) for x, y, _ in lines) >= 0.8:
                y_min = min(_value_at(0.9, x, y) for x, y, _ in lines)
                _add_zoom_inset(ax, lines, (0.8, 1.0), (y_min, 1.0), [0.3, 0.35, 0.4, 0.4])
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            title, legend_loc = "Macro-average Precision-Recall", 'lower left'

        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.02])
        ax.set_title(f"{title} - base models vs stacking / "
                     f"{split_name.upper()}", fontsize=13, fontweight='bold')
        ax.legend(loc=legend_loc, fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        print(f"\nModel-comparison {kind.upper()} overlay [split={split_name.upper()}]")
        return save_figure(fig, save_dir, f"{kind}_comparison_{self.MODEL_NAME}_{split_name}",
                           plot_format, dpi, show)


# ======================================================================
# Entry point
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run inference, evaluation and reporting on the JITrans '
                    'classifier or on the stacking meta-learner (rf / xgboost / catboost / logreg)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--model', type=str, default=None,
                        choices=['jit', 'stacking'],
                        help='Which model to evaluate (not needed with --train-log alone)')

    # --- JITrans: data / model ---
    parser.add_argument('--resume', '--model-path', dest='resume', type=str, default=None,
                        help='Path to the trained JITrans checkpoint (best.pt)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to the RSCD dataset root directory')
    parser.add_argument('--data-set', type=str, default='RSCD',
                        help='Dataset type understood by dataloader.build_dataset')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--patch-size', type=int, default=None,
                        help='Patch size (default: read from the checkpoint args)')
    parser.add_argument('--feedback-steps', type=int, default=None,
                        help='Decoder feedback steps (default: read from the checkpoint args)')
    parser.add_argument('--eval-crop-ratio', type=float, default=0.875,
                        help='Center-crop ratio used at evaluation time')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--amp', action='store_true',
                        help='Run the forward pass under autocast (CUDA only)')
    parser.add_argument('--strict', action='store_true',
                        help='Fail on missing/unexpected keys when loading the checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help='Enable gradient checkpointing in the model (matches training)')

    # --- stacking ---
    parser.add_argument('--meta-model', '--rf', dest='meta_model', type=str,
                        default='stacking/rf_meta.joblib',
                        help='Path to the meta-model saved by stacking.py '
                             '(<rf|xgboost|catboost|logreg>_meta.joblib; --rf is kept as an alias)')
    parser.add_argument('--preds-dir', type=str, default='stacking/preds',
                        help='Directory holding jit_*.npz and mamba_*.npz')
    parser.add_argument('--compare', action='store_true',
                        help='Compare JITrans, Mamba and the stacking side by side')

    # --- dataset split ---
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test', 'both', 'separate'],
                        help=(
                            'Which dataset split to evaluate. '
                            '"val" -> validation set. "test" -> test set. '
                            '"both" -> val + test as one dataset. '
                            '"separate" -> val AND test evaluated independently '
                            'and reported side by side.'
                        ))
    parser.add_argument('--class-names', type=str, nargs='+', default=None,
                        help='Override class names (default: from the checkpoint / ImageFolder)')

    # --- device ---
    parser.add_argument('--gpu-id', type=int, default=None,
                        help='GPU ID to use (default: auto-select the freest one)')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'],
                        help='Force device type')

    # --- analysis options ---
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate on the selected dataset split(s)')
    parser.add_argument('--model-summary', action='store_true',
                        help='Model complexity (params / FLOPs; for stacking: meta-learner size, '
                             'ops per sample and full-pipeline totals), printed and '
                             'saved as model_summary_<model>.json/.txt')
    parser.add_argument('--benchmark', action='store_true',
                        help='Latency benchmark (JITrans: forward on a synthetic '
                             'input; stacking: single-row predict_proba)')
    parser.add_argument('--memory', action='store_true',
                        help='Memory profiling (JITrans only)')
    parser.add_argument('--inference', action='store_true',
                        help='Single-image (batch=1) inference latency on real '
                             'images, saved as single_inference_<model>.json/.txt/.csv')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of images for --inference')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warm-up runs excluded from the --inference timings')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for the random image pick of --inference')
    parser.add_argument('--base-files', type=str, nargs='+', default=None,
                        help='Stacking: model_summary_<model>.json / '
                             'single_inference_<model>.json of JITrans and Mamba '
                             '(also searched automatically in --export-dir, '
                             '--preds-dir and next to --meta-model)')

    # --- plotting ---
    parser.add_argument('--plot-format', type=str, default='pdf',
                        choices=list(PLOT_FORMATS),
                        help='Output format for every figure')
    parser.add_argument('--plot-dir', type=str, default=None,
                        help='Directory where plots are saved (default: folder of the checkpoint)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Raster resolution (PNG only)')
    parser.add_argument('--show', action='store_true',
                        help='Display figures interactively (needs a display)')
    parser.add_argument('--plot-all', action='store_true',
                        help='Produce every evaluation plot')
    parser.add_argument('--plot-curves', action='store_true',
                        help='Plot AUC-ROC and AUC-PR curves')
    parser.add_argument('--plot-confusion-matrix', action='store_true',
                        help='Plot the confusion matrix (raw counts + row-normalized)')
    parser.add_argument('--plot-per-class', action='store_true',
                        help='Plot per-class metric bars and class distribution')
    parser.add_argument('--plot-confidence', action='store_true',
                        help='Plot confidence histograms and selective-prediction curves')
    parser.add_argument('--plot-calibration', action='store_true',
                        help='Plot the reliability diagram (ECE / MCE)')
    parser.add_argument('--plot-misclassified', action='store_true',
                        help='Plot a grid of the most confident misclassifications')
    parser.add_argument('--num-misclassified', type=int, default=16,
                        help='How many misclassified images to show')

    # --- training curves ---
    parser.add_argument('--train-log', type=str, nargs='+', default=None,
                        help=('Training log file(s), glob, or an output/<run>/ directory. '
                              'Plots loss, accuracy, F1 and confidence across epochs. '
                              'Needs neither the model nor the dataset.'))
    parser.add_argument('--log-scale-loss', action='store_true',
                        help='Use a logarithmic y-axis for the loss curve')

    # --- export ---
    parser.add_argument('--export-dir', type=str, default=None,
                        help='Directory for JSON/CSV/TXT metric exports (default: --plot-dir)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Also export per-sample predictions as CSV (can be large)')

    return parser


def get_best_device(force_cpu: bool = False, gpu_id: Optional[int] = None):
    """Pick a device; with several GPUs, the one with the most free memory."""
    require_eval_deps()
    if force_cpu:
        print(f"{WARN} Forcing CPU usage")
        return torch.device('cpu')

    if not torch.cuda.is_available():
        print(f"{WARN} CUDA not available, falling back to CPU")
        return torch.device('cpu')

    num_gpus = torch.cuda.device_count()

    if gpu_id is not None:
        if gpu_id >= num_gpus:
            raise ValueError(
                f"GPU {gpu_id} not available. Available GPUs: 0-{num_gpus-1}")
        print(f"{OK} Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        return torch.device(f'cuda:{gpu_id}')

    best_gpu, max_free_mem = None, -1
    for i in range(num_gpus):
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} | "
              f"Free: {free_mem/1024**3:.2f} GB / {total_mem/1024**3:.2f} GB")
        if free_mem > max_free_mem:
            max_free_mem, best_gpu = free_mem, i

    print(f"{OK} Auto-selected GPU {best_gpu}")
    return torch.device(f'cuda:{best_gpu}')


def resolve_device(args):
    """Pick the device from the CLI flags."""
    return get_best_device(force_cpu=(args.device == 'cpu'), gpu_id=args.gpu_id)


def _run_selected_plots(evaluator, results, args, plot_dir):
    """Dispatch the individual --plot-* flags for one split."""
    if args.plot_all:
        evaluator.plot_all(
            results, save_dir=plot_dir, show=args.show, dpi=args.dpi,
            plot_format=args.plot_format,
            top_n_misclassified=args.num_misclassified,
        )
        return

    if args.plot_curves:
        evaluator.plot_curves(results, plot_dir, args.show, args.dpi, args.plot_format)
    if args.plot_confusion_matrix:
        evaluator.plot_confusion_matrix(results, plot_dir, args.show, args.dpi,
                                        args.plot_format)
    if args.plot_per_class:
        evaluator.plot_per_class_metrics(results, plot_dir, args.show, args.dpi,
                                         args.plot_format)
        evaluator.plot_class_distribution(results, plot_dir, args.show, args.dpi,
                                          args.plot_format)
    if args.plot_confidence:
        evaluator.plot_confidence_histogram(results, plot_dir, args.show, args.dpi,
                                            args.plot_format)
    if args.plot_calibration:
        evaluator.plot_calibration(results, plot_dir, args.show, args.dpi,
                                   args.plot_format)
    if args.plot_misclassified:
        evaluator.plot_misclassified(results, plot_dir, args.show, args.dpi,
                                     args.plot_format, args.num_misclassified)


def main():
    """Command-line entry point."""
    parser = build_parser()
    args = parser.parse_args()

    wants_model = any([args.evaluate, args.model_summary, args.benchmark,
                       args.memory, args.compare, args.inference])
    # Evaluation-only plot flags imply --evaluate
    wants_eval_plots = any([
        args.plot_all, args.plot_curves, args.plot_confusion_matrix,
        args.plot_per_class, args.plot_confidence, args.plot_calibration,
        args.plot_misclassified,
    ])

    if not wants_model and not wants_eval_plots and not args.train_log:
        parser.error(
            "Nothing to do. Pick at least one action: --evaluate, --model-summary, "
            "--benchmark, --memory, --inference, --compare, a --plot-* flag, "
            "or --train-log."
        )

    if wants_eval_plots and not args.evaluate:
        print("Plot flags require an evaluation - enabling --evaluate")
        args.evaluate = True
        wants_model = True

    if wants_model and not args.model:
        parser.error("--model {jit,stacking} is required for the requested action(s).")

    if wants_model and args.model == 'jit':
        missing = [name for name, value in (('--resume', args.resume),
                                            ('--data-path', args.data_path))
                   if not value]
        if missing:
            parser.error(
                f"{', '.join(missing)} required for --model jit. "
                "Use only --train-log if you just want the training curves.")

    if args.compare and args.model != 'stacking':
        parser.error("--compare is only available with --model stacking.")

    # Default output directories: next to the checkpoint / meta-model
    plot_dir = args.plot_dir
    if plot_dir is None:
        anchor = args.resume if args.model == 'jit' else (
            args.meta_model if args.model == 'stacking' else None)
        if anchor:
            plot_dir = os.path.dirname(os.path.abspath(anchor))
        elif args.train_log:
            first = JITLogParser.resolve_paths(args.train_log, verbose=False)
            if first:
                plot_dir = os.path.dirname(os.path.abspath(first[0]))
    export_dir = args.export_dir or plot_dir

    # -- Training curves from the log (no model or dataset needed) -----
    if args.train_log:
        try:
            history = JITLogParser.parse(args.train_log, verbose=True)
            plot_training_curves(
                history,
                save_dir=plot_dir,
                plot_format=args.plot_format,
                show=args.show,
                dpi=args.dpi,
                log_scale=args.log_scale_loss,
            )
            if export_dir:
                history.to_csv(os.path.join(export_dir, 'training_history.csv'))
        except (FileNotFoundError, ValueError) as exc:
            print(f"{WARN} Training curves skipped: {exc}")

    if not wants_model:
        print("\nDone.")
        return

    # -- Build the evaluator -------------------------------------------
    try:
        evaluator = _build_evaluator(args, export_dir)
    except ImportError as exc:
        # Missing torch / scikit-learn: a one-line message beats a traceback.
        print(f"\n{WARN} {exc}")
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"\n{WARN} {exc}")
        sys.exit(1)

    # -- Model analysis (complexity / benchmark / memory) ---------------
    if args.model_summary or args.benchmark or args.memory:
        summary = evaluator.model_summary(
            include_benchmark=args.benchmark,
            include_memory=args.memory,
        )
        if export_dir:
            evaluator.export_model_summary(summary, export_dir)

    # -- Evaluate -------------------------------------------------------
    if args.evaluate:
        results_by_split = evaluator.evaluate_splits(
            verbose=True,
            save_predictions=args.save_predictions,
        )
        print(f"\n{OK} Evaluation complete!")

        for split_name, results in results_by_split.items():
            _run_selected_plots(evaluator, results, args, plot_dir)
            if export_dir:
                evaluator.export_results(results, export_dir,
                                         save_predictions=args.save_predictions)

        if len(results_by_split) > 1 and export_dir:
            evaluator.print_split_comparison(results_by_split, out_dir=export_dir)

    # -- Three-way model comparison ------------------------------------
    if args.compare:
        for split_name in evaluator.split_names:
            evaluator.compare_models(
                split_name,
                out_dir=export_dir,
                save_dir=plot_dir,
                show=args.show,
                dpi=args.dpi,
                plot_format=args.plot_format,
            )

    # -- Single-image inference latency ---------------------------------
    if args.inference:
        summary = evaluator.single_image_inference(
            num_samples=args.num_samples,
            warmup_runs=args.warmup,
            seed=args.seed,
        )
        if export_dir:
            evaluator.export_single_inference(summary, export_dir)

    print("\nDone.")


def _build_evaluator(args, export_dir: Optional[str] = None):
    """Instantiate the evaluator selected by --model."""
    if args.model == 'jit':
        device = resolve_device(args)
        print(f"\nInitializing JITrans inference (split={args.split})...")
        evaluator = JITInference(
            model_path=args.resume,
            data_path=args.data_path,
            data_set=args.data_set,
            img_size=args.img_size,
            patch_size=args.patch_size,
            feedback_steps=args.feedback_steps,
            eval_crop_ratio=args.eval_crop_ratio,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            split=args.split,
            class_names=args.class_names,
            amp=args.amp,
            strict=args.strict,
            use_checkpoint=args.use_checkpoint,
        )
        return evaluator

    print(f"\nInitializing stacking inference (split={args.split})...")
    return StackingInference(
        meta_path=args.meta_model,
        preds_dir=args.preds_dir,
        split=args.split,
        data_path=args.data_path,
        data_set=args.data_set,
        img_size=args.img_size,
        eval_crop_ratio=args.eval_crop_ratio,
        class_names=args.class_names,
        base_files=args.base_files,
        base_search_dirs=[export_dir] if export_dir else None,
    )


if __name__ == "__main__":
    main()
