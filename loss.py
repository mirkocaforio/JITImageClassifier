from abc import ABC, abstractmethod
from enum import Enum

import torch

try:
    from torchvision.ops.focal_loss import sigmoid_focal_loss
except ImportError:
    from torchvision.ops import sigmoid_focal_loss

import statistics

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from torch import nn


class LossType(Enum):
    """
    Available loss types
    """

    CROSSENTROPY = "crossentropy"
    BCE = "bce"
    MSE = "mse"
    FOCAL = "focal"


class LossStrategy(ABC):
    """Abstract base class for loss strategies"""

    @abstractmethod
    def compute_loss(self, outputs, targets):
        """Compute the loss"""

    @abstractmethod
    def compute_metrics(self, outputs, targets):
        """Compute evaluation metrics"""

    @abstractmethod
    def prepare_predictions(self, outputs):
        """Prepare predictions for metric calculation"""


class CrossEntropyLossStrategy(LossStrategy):
    """Strategy for Cross Entropy Loss (Multi-class classification)"""

    def __init__(self, num_classes=None, class_names=None, top_k=(1, 5),
                 label_smoothing=0.1, class_weights=None):
        """
        Args:
            num_classes: Number of classes (optional, auto-detected if None)
            class_names: List of class names for detailed reporting
            top_k: Tuple of k values for top-k accuracy (e.g., (1, 5) for top-1 and top-5)
            label_smoothing: Label smoothing of the cross entropy loss
            class_weights: Optional per-class weight tensor (on the training device)
        """
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.num_classes = num_classes
        self.class_names = class_names
        self.top_k = top_k

        # For tracking per-class metrics over time
        self.class_correct = None
        self.class_total = None

    def compute_loss(self, outputs, targets):
        return self.criterion(outputs, targets.long())

    def compute_metrics(self, outputs, targets, detailed=False):
        """
        Compute comprehensive metrics for multi-class classification

        Args:
            outputs: Model outputs (logits)
            targets: Ground truth labels
            detailed: If True, include per-class metrics and confusion matrix

        Returns:
            Dictionary of metrics
        """
        preds = self.prepare_predictions(outputs)
        targets_np = targets.long().cpu().numpy()

        # Get probabilities for additional metrics
        probs = torch.softmax(outputs, dim=1).cpu().numpy()

        # Auto-detect number of classes if not set
        if self.num_classes is None:
            self.num_classes = outputs.shape[1]

        # Basic metrics
        metrics = {
            # Accuracy metrics
            "accuracy": accuracy_score(targets_np, preds),
            "balanced_accuracy": balanced_accuracy_score(targets_np, preds),
            # F1 scores
            "f1_macro": f1_score(targets_np, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(
                targets_np, preds, average="weighted", zero_division=0
            ),
            "f1_micro": f1_score(targets_np, preds, average="micro", zero_division=0),
            # Precision and Recall
            "precision_macro": precision_score(
                targets_np, preds, average="macro", zero_division=0
            ),
            "precision_weighted": precision_score(
                targets_np, preds, average="weighted", zero_division=0
            ),
            "recall_macro": recall_score(
                targets_np, preds, average="macro", zero_division=0
            ),
            "recall_weighted": recall_score(
                targets_np, preds, average="weighted", zero_division=0
            ),
            # Agreement metrics
            "cohen_kappa": cohen_kappa_score(targets_np, preds),
            "matthews_corrcoef": matthews_corrcoef(targets_np, preds),
        }

        # Threshold-free metrics for binary problems (class 1 = positive)
        if self.num_classes == 2:
            metrics["f1_binary"] = f1_score(targets_np, preds, average="binary", zero_division=0)
            try:
                metrics["auroc"] = roc_auc_score(targets_np, probs[:, 1])
                metrics["auprc"] = average_precision_score(targets_np, probs[:, 1])
            except ValueError:
                # Only one class present in targets
                metrics["auroc"] = float("nan")
                metrics["auprc"] = float("nan")

        # Top-K accuracy
        for k in self.top_k:
            if k <= self.num_classes:
                try:
                    top_k_acc = top_k_accuracy_score(
                        targets_np, probs, k=k, labels=range(self.num_classes)
                    )
                    metrics[f"top_{k}_accuracy"] = top_k_acc
                except:
                    # Fallback if sklearn version doesn't support top_k_accuracy_score
                    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
                    top_k_acc = np.mean(
                        [
                            targets_np[i] in top_k_preds[i]
                            for i in range(len(targets_np))
                        ]
                    )
                    metrics[f"top_{k}_accuracy"] = top_k_acc

        # Confidence metrics
        max_probs = np.max(probs, axis=1)
        metrics["mean_confidence"] = np.mean(max_probs)
        metrics["median_confidence"] = np.median(max_probs)
        metrics["min_confidence"] = np.min(max_probs)

        # Correct prediction confidence
        correct_mask = preds == targets_np
        if correct_mask.sum() > 0:
            metrics["mean_correct_confidence"] = np.mean(max_probs[correct_mask])
        if (~correct_mask).sum() > 0:
            metrics["mean_incorrect_confidence"] = np.mean(max_probs[~correct_mask])

        # Per-class metrics (if requested)
        if detailed:
            metrics["per_class"] = self._compute_per_class_metrics(
                targets_np, preds, probs
            )

            # Confusion matrix
            metrics["confusion_matrix"] = confusion_matrix(
                targets_np, preds, labels=range(self.num_classes)
            )

        return metrics

    def _compute_per_class_metrics(self, targets, preds, probs):
        """Compute per-class precision, recall, f1, and support"""
        per_class = {}

        # Compute metrics for each class
        precision_per_class = precision_score(
            targets,
            preds,
            average=None,
            zero_division=0,
            labels=range(self.num_classes),
        )
        recall_per_class = recall_score(
            targets,
            preds,
            average=None,
            zero_division=0,
            labels=range(self.num_classes),
        )
        f1_per_class = f1_score(
            targets,
            preds,
            average=None,
            zero_division=0,
            labels=range(self.num_classes),
        )

        # Count samples per class
        unique, counts = np.unique(targets, return_counts=True)
        support_dict = dict(zip(unique, counts))

        for class_idx in range(self.num_classes):
            class_name = (
                self.class_names[class_idx]
                if self.class_names and class_idx < len(self.class_names)
                else f"class_{class_idx}"
            )

            per_class[class_name] = {
                "precision": float(precision_per_class[class_idx]),
                "recall": float(recall_per_class[class_idx]),
                "f1": float(f1_per_class[class_idx]),
                "support": int(support_dict.get(class_idx, 0)),
            }

            # Average confidence for this class
            class_mask = targets == class_idx
            if class_mask.sum() > 0:
                per_class[class_name]["mean_confidence"] = float(
                    np.mean(probs[class_mask, class_idx])
                )

        return per_class

    def prepare_predictions(self, outputs):
        """Convert logits to class predictions"""
        if outputs.shape[1] == 1:  # Binary case
            return (torch.sigmoid(outputs) > 0.5).long().squeeze().cpu().numpy()
        # Multi-class case
        return torch.argmax(outputs, dim=1).cpu().numpy()


class SigmoidFocalLossStrategy(LossStrategy):
    """Sigmoid Focal Loss for multi-label classification"""

    def __init__(self, num_classes=None, class_names=None, top_k=(1, 5), **kwargs):
        self.criterion = sigmoid_focal_loss
        self.average = "macro"  # Default average for metrics
        self.alpha = kwargs.get("alpha", 0.35)
        self.gamma = kwargs.get("gamma", 2.0)
        self.reduction = kwargs.get("reduction", "sum")
        self.multi_step = kwargs.get("multi_step", False)
        if self.multi_step:
            self.average = "samples"  # For multilabel

        self.num_classes = num_classes
        self.class_names = class_names
        self.top_k = top_k

        # For tracking per-class metrics over time
        self.class_correct = None
        self.class_total = None

        print(
            f"Initialized SigmoidFocalLoss with alpha={self.alpha}, gamma={self.gamma}, multi_step={self.multi_step}"
        )

    def compute_loss(self, outputs, targets, reduction="sum"):
        if self.multi_step:
            # Reshape outputs and targets for sequence classification
            B, T, C = outputs.shape
            outputs = outputs.view(-1, C)  # [B*T, C]
            targets = targets.view(-1, C)  # [B*T, C]

        if targets.dim() == 1:
            num_classes = outputs.shape[-1]
            targets = torch.nn.functional.one_hot(
                targets.long(), num_classes=num_classes
            ).float()

        if self.reduction == "sum":
            # Normalize by the number of positive samples to avoid large loss values
            num_positives = targets.sum().item()
            return self.criterion(
                outputs,
                targets.float(),
                alpha=self.alpha,
                gamma=self.gamma,
                reduction="sum",
            ) / max(num_positives, 1)
        else:
            return self.criterion(
                inputs=outputs,
                targets=targets.float(),
                alpha=self.alpha,
                gamma=self.gamma,
                reduction=self.reduction,
            )

    def compute_metrics(self, outputs, targets, detailed=False):
        """
        Compute comprehensive metrics for multi-class classification

        Args:
            outputs: Model outputs (logits)
            targets: Ground truth labels
            detailed: If True, include per-class metrics and confusion matrix

        Returns:
            Dictionary of metrics
        """
        preds = self.prepare_predictions(outputs)
        targets_np = self.prepare_targets(targets)

        # Get probabilities for additional metrics
        probs = torch.softmax(outputs, dim=1).cpu().numpy()

        # Auto-detect number of classes if not set
        if self.num_classes is None:
            self.num_classes = outputs.shape[1]

        # Basic metrics
        metrics = {
            # Accuracy metrics
            "accuracy": accuracy_score(targets_np, preds),
            "balanced_accuracy": balanced_accuracy_score(targets_np, preds),
            # F1 scores
            "f1_macro": f1_score(targets_np, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(
                targets_np, preds, average="weighted", zero_division=0
            ),
            "f1_micro": f1_score(targets_np, preds, average="micro", zero_division=0),
            # Precision and Recall
            "precision_macro": precision_score(
                targets_np, preds, average="macro", zero_division=0
            ),
            "precision_weighted": precision_score(
                targets_np, preds, average="weighted", zero_division=0
            ),
            "recall_macro": recall_score(
                targets_np, preds, average="macro", zero_division=0
            ),
            "recall_weighted": recall_score(
                targets_np, preds, average="weighted", zero_division=0
            ),
            # Agreement metrics
            "cohen_kappa": cohen_kappa_score(targets_np, preds),
            "matthews_corrcoef": matthews_corrcoef(targets_np, preds),
        }

        # Top-K accuracy
        for k in self.top_k:
            if k <= self.num_classes:
                try:
                    top_k_acc = top_k_accuracy_score(
                        targets_np, probs, k=k, labels=range(self.num_classes)
                    )
                    metrics[f"top_{k}_accuracy"] = top_k_acc
                except:
                    # Fallback if sklearn version doesn't support top_k_accuracy_score
                    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
                    top_k_acc = np.mean(
                        [
                            targets_np[i] in top_k_preds[i]
                            for i in range(len(targets_np))
                        ]
                    )
                    metrics[f"top_{k}_accuracy"] = top_k_acc

        # Confidence metrics
        max_probs = np.max(probs, axis=1)
        metrics["mean_confidence"] = np.mean(max_probs)
        metrics["median_confidence"] = np.median(max_probs)
        metrics["min_confidence"] = np.min(max_probs)

        # Correct prediction confidence
        correct_mask = preds == targets_np
        if correct_mask.sum() > 0:
            metrics["mean_correct_confidence"] = np.mean(max_probs[correct_mask])
        if (~correct_mask).sum() > 0:
            metrics["mean_incorrect_confidence"] = np.mean(max_probs[~correct_mask])

        # Per-class metrics (if requested)
        if detailed:
            metrics["per_class"] = self._compute_per_class_metrics(
                targets_np, preds, probs
            )

            # Confusion matrix
            metrics["confusion_matrix"] = confusion_matrix(
                targets_np, preds, labels=range(self.num_classes)
            )

        return metrics

    def _compute_per_class_metrics(self, targets, preds, probs):
        """Compute per-class precision, recall, f1, and support"""
        per_class = {}

        # Compute metrics for each class
        precision_per_class = precision_score(
            targets,
            preds,
            average=None,
            zero_division=0,
            labels=range(self.num_classes),
        )
        recall_per_class = recall_score(
            targets,
            preds,
            average=None,
            zero_division=0,
            labels=range(self.num_classes),
        )
        f1_per_class = f1_score(
            targets,
            preds,
            average=None,
            zero_division=0,
            labels=range(self.num_classes),
        )

        # Count samples per class
        unique, counts = np.unique(targets, return_counts=True)
        support_dict = dict(zip(unique, counts))

        for class_idx in range(self.num_classes):
            class_name = (
                self.class_names[class_idx]
                if self.class_names and class_idx < len(self.class_names)
                else f"class_{class_idx}"
            )

            per_class[class_name] = {
                "precision": float(precision_per_class[class_idx]),
                "recall": float(recall_per_class[class_idx]),
                "f1": float(f1_per_class[class_idx]),
                "support": int(support_dict.get(class_idx, 0)),
            }

            # Average confidence for this class
            class_mask = targets == class_idx
            if class_mask.sum() > 0:
                per_class[class_name]["mean_confidence"] = float(
                    np.mean(probs[class_mask, class_idx])
                )

        return per_class

    def prepare_predictions(self, outputs):
        """Convert logits to binary predictions"""
        if self.multi_step:
            # Per sequence: [B, T, C] → argmax sull'ultima dimensione
            return torch.argmax(outputs, dim=-1).detach().cpu().numpy()
        # Multi-class: prendi la classe con probabilità più alta
        return torch.argmax(outputs, dim=1).detach().cpu().numpy()

    def prepare_targets(self, targets):
        """Convert targets to numpy array"""
        if targets.dim() == 2:
            return targets.argmax(dim=1).cpu().numpy()
        else:
            return targets.long().cpu().numpy()


class LossStrategyFactory:
    """Factory for creating loss strategies"""

    @staticmethod
    def create_strategy(loss_type: str) -> LossStrategy:
        strategies = {
            "crossentropy": CrossEntropyLossStrategy,
            "focal": SigmoidFocalLossStrategy,
        }

        if loss_type.lower() not in strategies:
            raise ValueError(
                f"Unknown loss type: {loss_type}. Available: {list(strategies.keys())}"
            )

        return strategies[loss_type.lower()]()
