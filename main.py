#!/usr/bin/env python3

"""
Image Classification using the JITrans (Just In Time Transformers) Model

This script adapts the original autoregressive encoder-decoder JITrans architecture
(originally designed for time-series forecasting) to image classification, preserving
its autoregressive "feedback" (retroazione) form: the decoder refines its class
prediction over several steps, feeding each prediction back as input to the next step.

This source code is related to the paper "Just In Time Transformers", by
AHMED ALA EDDINE BENALI, MASSIMO CAFARO, ITALO EPICOCO, MARCO PULIMENO and ENRICO JUNIOR SCHIOPPA

The script parses command-line arguments, builds the RSCD image datasets, defines the
model, trains it with progress bars, saves the last/best checkpoints, supports resuming,
and reports detailed classification metrics on validation and test sets.

In order to train the model, the use of a recent NVIDIA gpu is highly recommended.

param_train.sh and jit_training.slurm launch the training on the RSCD dataset.

Example usage:
    python main.py --data_path ../datahome/RSCD_dataset_1million_reorganized/ --model_save_dir ./output/jitrans_run
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import build_dataset
from loss import CrossEntropyLossStrategy, LossStrategyFactory
from model import define_model

torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Logging setup  — writes to both stdout and <model_save_dir>/train.log
# ---------------------------------------------------------------------------


def setup_logger(log_path: str) -> logging.Logger:
    """
    Returns a logger that mirrors every message to stdout AND to log_path.
    Call this after model_save_dir is created.
    """
    logger = logging.getLogger("jitrans")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

SELECT_METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "auroc", "auprc"]


def get_args_parser():
    parser = argparse.ArgumentParser(
        "JITrans image classifier", add_help=True
    )

    # Data
    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="Image dataset root (must contain train/val/test)",
    )
    parser.add_argument("--data_set", default="RSCD", choices=["RSCD"], type=str)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--patch_size", default=16, type=int)

    # Optimisation
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--num_workers", default=4, type=int)

    # Model selection / early stopping
    parser.add_argument(
        "--select_metric",
        default="accuracy",
        choices=SELECT_METRICS,
        help="Validation metric for best checkpoint and early stopping",
    )
    parser.add_argument(
        "--early_stopping",
        default=20,
        type=int,
        help="Stop training after this many epochs with no improvement of the "
        "selection metric. Set to 0 to disable.",
    )

    # Model / training behaviour
    parser.add_argument("--feedback_steps", default=4, type=int)
    parser.add_argument("--loss_type", default="crossentropy", type=str)
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--deep_supervision", action="store_true")
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--amp", action="store_true")

    # Checkpointing / output
    parser.add_argument("--model_save_dir", default="./output", type=str)
    parser.add_argument("--resume", default="", type=str)

    # Data augmentation / transform
    parser.add_argument("--color_jitter", default=0.4, type=float)
    parser.add_argument("--aa", default="rand-m9-mstd0.5-inc1", type=str)
    parser.add_argument("--train_interpolation", default="bicubic", type=str)
    parser.add_argument("--reprob", default=0.25, type=float)
    parser.add_argument("--remode", default="pixel", type=str)
    parser.add_argument("--recount", default=1, type=int)
    parser.add_argument("--eval_crop_ratio", default=0.875, type=float)

    return parser


# ---------------------------------------------------------------------------
# Loss strategy
# ---------------------------------------------------------------------------


def build_loss_strategy(args, num_classes, class_names, class_weights=None):
    if args.loss_type.lower() == "crossentropy":
        return CrossEntropyLossStrategy(
            num_classes=num_classes,
            class_names=class_names,
            top_k=(1, 5) if num_classes >= 5 else (1,),
            label_smoothing=args.label_smoothing,
            class_weights=class_weights,
        )
    strategy = LossStrategyFactory.create_strategy(args.loss_type)
    strategy.num_classes = num_classes
    strategy.class_names = class_names
    return strategy


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model, loader, strategy, optimizer, scaler, device, epoch, num_epochs, args, logger
):
    model.train()
    running_loss = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc=f"Train {epoch + 1}/{num_epochs}", leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            if args.deep_supervision:
                _, all_logits = model(images, return_all_steps=True)
                step_losses = [
                    strategy.compute_loss(all_logits[:, t], targets)
                    for t in range(all_logits.shape[1])
                ]
                loss = torch.stack(step_losses).mean()
            else:
                logits = model(images)
                loss = strategy.compute_loss(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(
            loss=f"{running_loss / n_batches:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return running_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model,
    loader,
    strategy,
    device,
    args,
    desc="Val",
    detailed=False,
    return_outputs=False,
):
    model.eval()
    all_logits = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets_dev = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            logits = model(images)
        loss = strategy.compute_loss(logits, targets_dev)
        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits.float().cpu())
        all_targets.append(targets.cpu())
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    avg_loss = total_loss / max(n_batches, 1)
    metrics = strategy.compute_metrics(all_logits, all_targets, detailed=detailed)
    metrics["loss"] = avg_loss  # inject loss into the metrics dict
    if return_outputs:
        return metrics, all_logits, all_targets
    return metrics


# ---------------------------------------------------------------------------
# Print / log helpers
# ---------------------------------------------------------------------------


def _log_block(lines, logger):
    """Send a list of strings to the logger line by line."""
    for line in lines:
        logger.info(line)


def _scalar_lines(metrics, W):
    lines = []
    for k in ["accuracy", "balanced_accuracy", "top_1_accuracy", "top_5_accuracy"]:
        if k in metrics:
            lines.append(f"  {k:<34}: {metrics[k]:.4f}")
    lines.append(f"  {'-' * (W - 2)}")
    for k in ["auroc", "auprc"]:
        if k in metrics:
            lines.append(f"  {k:<34}: {metrics[k]:.4f}")
    for k in ["f1_binary", "f1_macro", "f1_weighted", "f1_micro"]:
        if k in metrics:
            lines.append(f"  {k:<34}: {metrics[k]:.4f}")
    lines.append(f"  {'-' * (W - 2)}")
    for k in [
        "precision_macro",
        "precision_weighted",
        "recall_macro",
        "recall_weighted",
    ]:
        if k in metrics:
            lines.append(f"  {k:<34}: {metrics[k]:.4f}")
    lines.append(f"  {'-' * (W - 2)}")
    for k in ["cohen_kappa", "matthews_corrcoef"]:
        if k in metrics:
            lines.append(f"  {k:<34}: {metrics[k]:.4f}")
    return lines


def _confidence_lines(metrics, W):
    keys = [
        "mean_confidence",
        "median_confidence",
        "min_confidence",
        "mean_correct_confidence",
        "mean_incorrect_confidence",
    ]
    if not any(k in metrics for k in keys):
        return []
    lines = [f"  {'-' * (W - 2)}"]
    for k in keys:
        if k in metrics:
            lines.append(f"  {k:<34}: {metrics[k]:.4f}")
    return lines


def _per_class_lines(metrics, W):
    if "per_class" not in metrics:
        return []
    lines = [
        "",
        "  Per-class breakdown",
        f"  {'Class':<30}  Prec   Rec    F1    Support",
        f"  {'-' * (W - 2)}",
    ]
    for cls_name, m in metrics["per_class"].items():
        conf = f"  conf={m['mean_confidence']:.3f}" if "mean_confidence" in m else ""
        lines.append(
            f"  {cls_name:<30}"
            f"  {m['precision']:.3f}"
            f"  {m['recall']:.3f}"
            f"  {m['f1']:.3f}"
            f"  {m['support']:>7}"
            f"{conf}"
        )
    return lines


def _confusion_lines(metrics, class_names, W):
    if "confusion_matrix" not in metrics:
        return []
    cm = metrics["confusion_matrix"]
    lines = ["", "  Confusion matrix (rows=true, cols=pred):"]
    header = "    " + "".join(f"{i:>8d}" for i in range(cm.shape[1]))
    lines.append(header)
    for i, row in enumerate(cm):
        lines.append(f"    {i:<4d}" + "".join(f"{int(v):>8d}" for v in row))
    if class_names:
        lines.append("")
        lines.append("  Class index legend:")
        for i, name in enumerate(class_names):
            lines.append(f"    {i}: {name}")
    return lines


def print_best_report(metrics, epoch, class_names, logger):
    """Detailed report printed (and logged) whenever a new best is reached."""
    W = 62
    lines = [
        "",
        "★" * W,
        f"  NEW BEST  —  Epoch {epoch + 1}",
        "★" * W,
        f"  {'Val Loss':<34}: {metrics.get('loss', float('nan')):.4f}",
        f"  {'-' * (W - 2)}",
    ]
    lines += _scalar_lines(metrics, W)
    lines += _confidence_lines(metrics, W)
    lines += _per_class_lines(metrics, W)
    lines += _confusion_lines(metrics, class_names, W)
    lines += ["★" * W, ""]
    _log_block(lines, logger)


def print_final_report(metrics, class_names, logger):
    """Full report used only at end-of-training test evaluation."""
    W = 62
    lines = [
        "",
        "=" * W,
        "  TEST — FINAL RESULTS",
        "=" * W,
        f"  {'Test Loss':<34}: {metrics.get('loss', float('nan')):.4f}",
        f"  {'-' * (W - 2)}",
    ]
    lines += _scalar_lines(metrics, W)
    lines += _confidence_lines(metrics, W)
    lines += _per_class_lines(metrics, W)
    lines += _confusion_lines(metrics, class_names, W)
    lines += ["=" * W, ""]
    _log_block(lines, logger)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    path, model, optimizer, scaler, epoch, best_acc, args, class_names, extra=None
):
    """`best_acc` holds the best value of args.select_metric."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_acc": best_acc,
            "args": vars(args),
            "class_names": class_names,
            **(extra or {}),
        },
        path,
    )


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


def main():
    args = get_args_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.model_save_dir, exist_ok=True)

    # Logger — mirrors to stdout and to train.log in the output directory
    log_path = os.path.join(args.model_save_dir, "train.log")
    logger = setup_logger(log_path)

    logger.info(f"Using device: {device}")
    logger.info(f"Log file    : {log_path}")

    if args.img_size % args.patch_size != 0:
        raise ValueError(
            f"img_size ({args.img_size}) must be divisible by patch_size ({args.patch_size})."
        )

    # --- Datasets / dataloaders ---
    train_dataset, nb_classes = build_dataset(is_train=True, is_test=False, args=args)
    val_dataset, _ = build_dataset(is_train=False, is_test=False, args=args)
    test_dataset, _ = build_dataset(is_train=False, is_test=True, args=args)
    class_names = list(
        getattr(train_dataset, "classes", [str(i) for i in range(nb_classes)])
    )

    logger.info(f"Classes ({nb_classes}): {class_names}")
    logger.info(
        f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}"
    )

    if args.select_metric in ("auroc", "auprc") and nb_classes != 2:
        raise ValueError(
            f"--select_metric {args.select_metric} needs a binary task "
            f"({nb_classes} classes found)."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- Model / loss / optimiser ---
    model = define_model(
        device,
        num_classes=nb_classes,
        patch_size=args.patch_size,
        img_size=args.img_size,
        feedback_steps=args.feedback_steps,
        use_checkpoint=args.use_checkpoint,
    )
    strategy = build_loss_strategy(args, nb_classes, class_names)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    logger.info(
        f"Epochs: {args.num_epochs} | Batch: {args.batch_size} | LR: {args.lr} | "
        f"Selection metric: {args.select_metric}"
    )

    # --- Resume ---
    start_epoch = 0
    best_acc = 0.0  # best value of args.select_metric
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_acc = ckpt.get("best_acc", 0.0)
        start_epoch = ckpt.get("epoch", -1) + 1
        logger.info(f"Resumed at epoch {start_epoch} (best_acc={best_acc:.4f}).")

    last_path = os.path.join(args.model_save_dir, "last.pt")
    best_path = os.path.join(args.model_save_dir, "best.pt")
    metric_label = "acc" if args.select_metric == "accuracy" else args.select_metric

    # Early stopping state
    es_patience = args.early_stopping  # 0 = disabled
    epochs_no_imp = 0

    # --- Training loop ---
    for epoch in range(start_epoch, args.num_epochs):
        # Training pass
        train_loss = train_one_epoch(
            model,
            train_loader,
            strategy,
            optimizer,
            scaler,
            device,
            epoch,
            args.num_epochs,
            args,
            logger,
        )

        # Validation pass (fast — no per-class detail)
        val_metrics = evaluate(
            model,
            val_loader,
            strategy,
            device,
            args,
            desc=f"Val {epoch + 1}/{args.num_epochs}",
            detailed=False,
        )
        val_acc = val_metrics["accuracy"]
        val_loss = val_metrics["loss"]
        val_score = val_metrics[args.select_metric]

        # One-line epoch summary
        is_best = val_score > best_acc
        best_marker = "  ★ new best" if is_best else ""
        extra_score = (
            ""
            if args.select_metric == "accuracy"
            else f"  {metric_label}: {val_score:.4f}"
        )
        logger.info(
            f"Epoch {epoch + 1}/{args.num_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f}  "
            f"f1: {val_metrics.get('f1_macro', 0):.4f}"
            f"{extra_score}"
            f"{best_marker}"
        )

        # Update the best value before saving last.pt, so a resume restarts from
        # the true best and cannot overwrite best.pt with a worse epoch
        if is_best:
            best_acc = val_score
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1

        # Always save last checkpoint
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scaler,
            epoch,
            best_acc,
            args,
            class_names,
        )

        # New best → save best.pt + detailed validation report
        if is_best:
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scaler,
                epoch,
                best_acc,
                args,
                class_names,
            )
            logger.info(
                f"  Best model saved to {best_path}  (val {metric_label}={best_acc:.4f})"
            )

            # Re-evaluate with detailed=True for the full breakdown
            val_metrics_det = evaluate(
                model,
                val_loader,
                strategy,
                device,
                args,
                desc=f"  Val (detailed) {epoch + 1}",
                detailed=True,
            )
            val_metrics_det["loss"] = val_loss  # carry the already-computed loss
            print_best_report(val_metrics_det, epoch, class_names, logger)

        # Early stopping check
        if es_patience > 0 and epochs_no_imp >= es_patience:
            logger.info(
                f"Early stopping triggered: no improvement for {es_patience} consecutive epochs."
            )
            break

    logger.info(f"\nTraining complete.")
    logger.info(f"  Last checkpoint : {last_path}")
    logger.info(f"  Best model      : {best_path}  (val {metric_label}={best_acc:.4f})")

    # --- Final evaluation on the test set using the best checkpoint ---
    if os.path.isfile(best_path):
        logger.info(f"\nLoading best checkpoint for final test: {best_path}")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            f"  (best was epoch {ckpt['epoch'] + 1}, val {metric_label}={ckpt['best_acc']:.4f})"
        )

    test_metrics = evaluate(
        model,
        test_loader,
        strategy,
        device,
        args,
        desc="Test",
        detailed=True,
    )
    print_final_report(test_metrics, class_names, logger)

    # Persist test metrics as JSON
    metrics_path = os.path.join(args.model_save_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(to_serializable(test_metrics), f, indent=2)
    logger.info(f"Test metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
