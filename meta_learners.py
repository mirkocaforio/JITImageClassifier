#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Meta-learners of the stacking ensemble: RandomForest, XGBoost and CatBoost.

Everything that depends on the kind of meta-learner lives here, so that `stacking.py`
(training) and `inference.py` (evaluation / complexity / latency) stay agnostic:

  - build_meta_learner / fit_meta_learner   construction and training, including the
                                            class balancing of each library
  - predict_proba_canonical                 probabilities in the canonical 0..C-1 order,
                                            optionally on a single thread (latency runs)
  - tree_ensemble_stats                     size and per-sample cost of the tree ensemble

xgboost and catboost are imported lazily, so the RandomForest path works without them.
"""

import contextlib

import numpy as np

META_LEARNERS = ("rf", "xgboost", "catboost")
DISPLAY_NAMES = {"rf": "RandomForest", "xgboost": "XGBoost", "catboost": "CatBoost"}
CLASS_WEIGHTS = ("balanced", "balanced_subsample", "none")

# Tree depth used by the boosting models when --max_depth is not given
# (an unlimited depth is only meaningful for the RandomForest)
DEFAULT_BOOSTING_DEPTH = 6


def display_name(name):
    return DISPLAY_NAMES.get(name, name)


def meta_learner_name(model):
    """Short name ('rf' / 'xgboost' / 'catboost') of a fitted estimator."""
    cls = type(model).__name__
    if cls == "RandomForestClassifier":
        return "rf"
    if cls == "XGBClassifier":
        return "xgboost"
    if cls == "CatBoostClassifier":
        return "catboost"
    raise ValueError(f"Unsupported meta-learner: {cls}")


# ---------------------------------------------------------------------------
# Construction / training
# ---------------------------------------------------------------------------

def build_meta_learner(
    name,
    n_classes,
    n_estimators=300,
    max_depth=None,
    learning_rate=0.1,
    min_samples_leaf=1,
    subsample=1.0,
    colsample_bytree=1.0,
    l2_leaf_reg=3.0,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
):
    """
    Instantiate an (unfitted) meta-learner.
    Args:
        name (str): 'rf', 'xgboost' or 'catboost'.
        n_classes (int): Number of classes of the task.
        n_estimators (int): Trees of the forest / boosting rounds.
        max_depth (int | None): Tree depth. None = unlimited for rf, 6 for the boosting models.
        learning_rate (float): Shrinkage of the boosting models (ignored by rf).
        min_samples_leaf (int): rf only.
        subsample, colsample_bytree (float): xgboost only (row / column sampling per tree).
        l2_leaf_reg (float): catboost only (L2 regularisation of the leaf values).
        class_weight (str): 'balanced', 'balanced_subsample' (rf only) or 'none'.
        random_state (int): Seed.
        n_jobs (int): Threads (-1 = all cores).
    Returns:
        The estimator. Train it with `fit_meta_learner`, which applies the class weighting
        of the libraries that take it at fit time (xgboost).
    """
    if name not in META_LEARNERS:
        raise ValueError(f"Unknown meta-learner '{name}', choose from {META_LEARNERS}")
    class_weight = class_weight.lower()
    if class_weight not in CLASS_WEIGHTS:
        raise ValueError(f"class_weight must be one of {CLASS_WEIGHTS}, got '{class_weight}'")
    if class_weight == "balanced_subsample" and name != "rf":
        raise ValueError("class_weight 'balanced_subsample' is available only for rf")

    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=None if class_weight == "none" else class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    depth = DEFAULT_BOOSTING_DEPTH if max_depth is None else max_depth

    if name == "xgboost":
        from xgboost import XGBClassifier

        # The objective (binary:logistic / multi:softprob) is chosen by fit() from the labels
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method="hist",
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
            random_state=random_state,
            n_jobs=n_jobs,
        )

    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=n_estimators,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        loss_function="Logloss" if n_classes == 2 else "MultiClass",
        auto_class_weights=None if class_weight == "none" else "Balanced",
        random_seed=random_state,
        thread_count=n_jobs,
        verbose=False,
        allow_writing_files=False,  # no catboost_info/ directory
    )


def check_labels(y, n_classes, split_name="val"):
    """Every class must appear in the meta-learner training labels (xgboost requires it)."""
    missing = sorted(set(range(n_classes)) - set(np.unique(y).tolist()))
    if missing:
        raise ValueError(
            f"[{split_name}] classes {missing} have no sample: the meta-learner cannot "
            f"learn them (labels must cover 0..{n_classes - 1})."
        )


def fit_meta_learner(model, X, y, class_weight="balanced"):
    """Fit the meta-learner; xgboost has no class_weight, so balancing goes through sample_weight."""
    if meta_learner_name(model) == "xgboost" and class_weight.lower() == "balanced":
        from sklearn.utils.class_weight import compute_sample_weight

        return model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return model.fit(X, y)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def n_threads(model):
    """Threads the meta-learner was trained with (n_jobs / thread_count)."""
    if meta_learner_name(model) == "catboost":
        # get_param is None when the default (-1, all cores) was kept
        threads = model.get_param("thread_count")
        return -1 if threads is None else threads
    return getattr(model, "n_jobs", None)


@contextlib.contextmanager
def single_thread(model):
    """
    Run the meta-learner on one thread, yielding the thread setting it was trained with.

    With n_jobs = -1 every predict_proba on one row pays the thread pool start-up, which
    dwarfs the actual tree traversal. CatBoost takes the thread count as a predict_proba
    argument instead (see predict_proba_canonical), so it is left untouched here.
    """
    name = meta_learner_name(model)
    original = n_threads(model)
    if name == "rf":
        model.n_jobs = 1
    elif name == "xgboost":
        model.set_params(n_jobs=1)
    try:
        yield original
    finally:
        if name == "rf":
            model.n_jobs = original
        elif name == "xgboost":
            model.set_params(n_jobs=original)


def raw_predict_proba(model, X, single_thread_predict=False):
    """predict_proba with the columns in model.classes_ order."""
    if single_thread_predict and meta_learner_name(model) == "catboost":
        return model.predict_proba(X, thread_count=1)
    return model.predict_proba(X)


def reorder_proba(model, proba_raw, n_classes):
    """Map the model.classes_ columns onto the canonical 0..C-1 class order."""
    proba = np.zeros((proba_raw.shape[0], n_classes), dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        proba[:, int(cls)] = proba_raw[:, col]
    return proba


def predict_proba_canonical(model, X, n_classes, single_thread_predict=False):
    """Class probabilities (N, n_classes) in the canonical 0..C-1 order."""
    return reorder_proba(model, raw_predict_proba(model, X, single_thread_predict), n_classes)


# ---------------------------------------------------------------------------
# Complexity
# ---------------------------------------------------------------------------

def tree_ensemble_stats(model, X):
    """
    Size and per-sample cost of the tree ensemble, measured on the meta-features X.

    Returned keys: n_trees, total_nodes, total_split_nodes, total_leaves,
    mean_nodes_per_tree, max_depth, mean_depth, stored_values, comparisons_per_sample,
    additions_per_sample, max_depth_param, cost_convention.

    comparisons_per_sample is the tree-ensemble analogue of FLOPs; it is NOT the same unit.
    """
    name = meta_learner_name(model)
    if name == "rf":
        return _rf_stats(model, X)
    if name == "xgboost":
        return _xgb_stats(model, X)
    return _catboost_stats(model, X)


def _rf_stats(rf, X):
    trees = [est.tree_ for est in rf.estimators_]
    node_counts = np.array([t.node_count for t in trees])
    leaves = np.array([t.n_leaves for t in trees])
    depths = np.array([t.max_depth for t in trees])
    n_classes = len(rf.classes_)
    split_nodes = node_counts - leaves

    indicator, _ = rf.decision_path(X)
    nodes_per_sample = indicator.nnz / X.shape[0]
    return {
        "n_trees": len(trees),
        "total_nodes": int(node_counts.sum()),
        "total_split_nodes": int(split_nodes.sum()),
        "total_leaves": int(leaves.sum()),
        "mean_nodes_per_tree": float(node_counts.mean()),
        "max_depth": int(depths.max()),
        "mean_depth": float(depths.mean()),
        # one feature index + one threshold per split node, one class distribution per leaf
        "stored_values": int(split_nodes.sum() * 2 + leaves.sum() * n_classes),
        "comparisons_per_sample": float(nodes_per_sample - len(trees)),  # a leaf does not compare
        "additions_per_sample": float(len(trees) * n_classes),
        "max_depth_param": rf.max_depth,
        "cost_convention": "split comparisons (measured with decision_path) + "
                           "n_trees x n_classes probability additions",
    }


def _xgb_stats(model, X):
    import xgboost

    booster = model.get_booster()
    df = booster.trees_to_dataframe()

    # Depth of every node (indexed by node id), walking each tree from its root through
    # the Yes/No children; trees are listed in booster order
    node_depths = []
    for tree_id, tree in df.groupby("Tree", sort=True):
        children = {
            int(row.Node): (int(row.Yes.split("-")[1]), int(row.No.split("-")[1]))
            for row in tree.itertuples() if row.Feature != "Leaf"
        }
        depth = np.zeros(int(tree["Node"].max()) + 1, dtype=np.int64)
        stack = [(0, 0)]
        while stack:
            node, d = stack.pop()
            depth[node] = d
            for child in children.get(node, ()):
                stack.append((child, d + 1))
        node_depths.append(depth)

    is_leaf = df["Feature"] == "Leaf"
    leaves_per_tree = is_leaf.groupby(df["Tree"]).sum().to_numpy()
    nodes_per_tree = df.groupby("Tree").size().to_numpy()
    tree_depths = np.array([d.max() for d in node_depths])
    n_trees = len(node_depths)
    split_nodes = int(nodes_per_tree.sum() - leaves_per_tree.sum())

    # Leaf reached by every sample in every tree -> its depth = comparisons in that tree
    leaf_ids = booster.predict(xgboost.DMatrix(X), pred_leaf=True).astype(np.int64)
    leaf_ids = leaf_ids.reshape(X.shape[0], n_trees)
    comparisons = np.zeros(X.shape[0], dtype=np.float64)
    for col, depth in enumerate(node_depths):
        comparisons += depth[leaf_ids[:, col]]

    return {
        "n_trees": n_trees,
        "total_nodes": int(nodes_per_tree.sum()),
        "total_split_nodes": split_nodes,
        "total_leaves": int(leaves_per_tree.sum()),
        "mean_nodes_per_tree": float(nodes_per_tree.mean()),
        "max_depth": int(tree_depths.max()),
        "mean_depth": float(tree_depths.mean()),
        # one feature index + one threshold per split node, one scalar margin per leaf
        "stored_values": int(split_nodes * 2 + leaves_per_tree.sum()),
        "comparisons_per_sample": float(comparisons.mean()),
        "additions_per_sample": float(n_trees),
        "max_depth_param": model.max_depth,
        "cost_convention": "split comparisons (measured with pred_leaf) + one margin "
                           "addition per tree (n_rounds x n_classes trees for multiclass)",
    }


def _catboost_stats(model, X):
    # CatBoost grows oblivious (symmetric) trees: every level tests one condition shared by
    # all its nodes, so a sample always performs exactly `depth` comparisons per tree.
    leaf_counts = np.asarray(model.get_tree_leaf_counts(), dtype=np.int64)
    depths = np.log2(leaf_counts).round().astype(np.int64)
    n_leaf_values = int(np.asarray(model.get_leaf_values()).size)
    n_trees = int(model.tree_count_)
    split_conditions = int(depths.sum())
    approx_dim = n_leaf_values // max(int(leaf_counts.sum()), 1)

    return {
        "n_trees": n_trees,
        # nodes of the equivalent binary trees; only `depth` distinct conditions are stored
        "total_nodes": int((2 * leaf_counts - 1).sum()),
        "total_split_nodes": int((leaf_counts - 1).sum()),
        "total_leaves": int(leaf_counts.sum()),
        "mean_nodes_per_tree": float((2 * leaf_counts - 1).mean()),
        "max_depth": int(depths.max()),
        "mean_depth": float(depths.mean()),
        # one feature index + one border per level, approx_dim values per leaf
        "stored_values": int(split_conditions * 2 + n_leaf_values),
        "comparisons_per_sample": float(split_conditions),
        "additions_per_sample": float(n_trees * approx_dim),
        "max_depth_param": model.get_param("depth"),
        "cost_convention": "split comparisons (depth per oblivious tree, exact) + "
                           "n_trees x leaf-dimension additions",
    }
