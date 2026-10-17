#!/usr/bin/env python3

"""
Export per-sample softmax probabilities of the JITrans model for the stacking ensemble.

Runs in the JIT conda env. Loads a trained JITrans checkpoint and, for the `val` and `test`
splits of the RSCD dataset, saves a compressed .npz file containing, for every image:
  - probs     : (N, num_classes) float32   softmax probabilities
  - rel_paths : (N,)            str         "<class_folder>/<file_name>" (alignment key)
  - targets   : (N,)            int64       ground-truth class index
  - classes   : (num_classes,)  str         sorted ImageFolder class names

The rel_paths key lets `stacking.py` align JITrans and Mamba predictions per image,
independently of DataLoader ordering.
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import define_model
from dataloader import build_dataset


def get_args():
    p = argparse.ArgumentParser("Export JITrans softmax probabilities for stacking")
    # Data
    p.add_argument("--data_path", required=True, type=str)
    p.add_argument("--data_set", default="RSCD", type=str)
    p.add_argument("--img_size", default=224, type=int)
    p.add_argument("--patch_size", default=16, type=int)
    p.add_argument("--eval_crop_ratio", default=0.875, type=float)
    # Model / checkpoint
    p.add_argument(
        "--resume",
        required=True,
        type=str,
        help="Path to the trained JITrans checkpoint (best.pt).",
    )
    p.add_argument("--feedback_steps", default=4, type=int)
    # Runtime
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out_dir", default="./stacking/preds", type=str)
    return p.parse_args()


def rel_paths_from_dataset(dataset):
    """Return ['<class_folder>/<file_name>', ...] in dataset.samples order."""
    rels = []
    for path, _ in dataset.samples:
        cls = os.path.basename(os.path.dirname(path))
        fname = os.path.basename(path)
        rels.append(f"{cls}/{fname}")
    return rels


@torch.no_grad()
def export_split(model, dataset, device, args, out_path):
    # shuffle=False, drop_last=False -> batch order follows dataset.samples order exactly.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    model.eval()
    all_probs, all_targets = [], []
    for images, targets in tqdm(loader, desc=os.path.basename(out_path), leave=False):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            logits = model(images)
        probs = torch.softmax(logits.float(), dim=1)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.numpy())

    probs = np.concatenate(all_probs, axis=0).astype(np.float32)
    targets = np.concatenate(all_targets, axis=0).astype(np.int64)
    rel_paths = np.array(rel_paths_from_dataset(dataset), dtype=object)
    classes = np.array(list(dataset.classes), dtype=object)

    assert len(probs) == len(targets) == len(rel_paths), "Length mismatch in export."

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path, probs=probs, rel_paths=rel_paths, targets=targets, classes=classes
    )
    print(f"[export_probs] wrote {out_path}  (N={len(probs)}, classes={list(classes)})")


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[export_probs] device: {device}")

    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

    val_dataset, nb_classes = build_dataset(is_train=False, is_test=False, args=args)
    test_dataset, _ = build_dataset(is_train=False, is_test=True, args=args)
    print(f"[export_probs] classes ({nb_classes}): {val_dataset.classes}")
    print(f"[export_probs] val: {len(val_dataset)} | test: {len(test_dataset)}")

    model = define_model(
        device,
        num_classes=nb_classes,
        patch_size=args.patch_size,
        img_size=args.img_size,
        feedback_steps=args.feedback_steps,
    )
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state)
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        print(
            f"[export_probs] loaded checkpoint (epoch={ckpt.get('epoch')}, "
            f"best_acc={ckpt.get('best_acc')})"
        )

    export_split(
        model, val_dataset, device, args, os.path.join(args.out_dir, "jit_val.npz")
    )
    export_split(
        model, test_dataset, device, args, os.path.join(args.out_dir, "jit_test.npz")
    )


if __name__ == "__main__":
    main()
