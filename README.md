# JITImageClassifier

Image classification on the RSCD road-surface dataset with the JITrans (Just In Time
Transformers) encoder-decoder, plus a stacking ensemble of JITrans + RSCDMamba.

The image is split into patches that the encoder turns into a memory; the decoder refines
the class prediction autoregressively over `--feedback_steps` steps, feeding each predicted
class distribution back as its next input token.

## Contents

| File | Purpose |
|---|---|
| `main.py` | Training / validation / test of JITrans, checkpoints `last.pt` / `best.pt`, resume |
| `model.py` | JITrans encoder-decoder (`define_model`) |
| `loss.py` | Loss strategies and the metric set used everywhere |
| `dataloader.py` | RSCD `ImageFolder` datasets and timm transforms |
| `export_probs.py` | Exports JITrans softmax probabilities on val/test for the stacking |
| `stacking.py` | Trains the stacking meta-learner on val, evaluates on test |
| `meta_learners.py` | RandomForest / XGBoost / CatBoost / multinomial logistic regression meta-learners |
| `tune_meta.py` | Chooses the meta-learner hyper-parameters by cross-validation on val |
| `inference.py` | Evaluation, plots, complexity and latency of JITrans and of the stacking |
| `param_train.sh`, `jit_training.slurm` | Training launchers (local GPU / SLURM) |

## Installation

    pip install -r requirements.txt

## Training

    python main.py --data_path ../datahome/RSCD_dataset_1million_reorganized/ \
        --model_save_dir ./output/jitrans_run

The dataset root must contain `train/`, `val/` and `test/` in `ImageFolder` layout.
`./param_train.sh --help` lists the launcher options.

## Stacking ensemble

1. Export the base-model probabilities (JITrans here, Mamba in its own repo):

       python export_probs.py --data_path <RSCD> --resume output/<run>/best.pt --out_dir stacking/preds

   and copy `mamba_val.npz` / `mamba_test.npz` into the same directory.

2. Choose the hyper-parameters by stratified 5-fold cross-validation on val (test is not
   read; the best configuration has the lowest mean CV log-loss, and the boosting rounds
   come from early stopping inside the folds):

       python tune_meta.py --preds_dir stacking/preds --out_dir stacking --meta_learner logreg

   Prints the `stacking.py` command of the best configuration and writes
   `tune_<meta_learner>.json` with every configuration of the grid.

3. Train the meta-learner on val and evaluate it on test:

       python stacking.py --preds_dir stacking/preds --out_dir stacking --meta_learner rf
       python stacking.py --preds_dir stacking/preds --out_dir stacking --meta_learner xgboost
       python stacking.py --preds_dir stacking/preds --out_dir stacking --meta_learner catboost
       python stacking.py --preds_dir stacking/preds --out_dir stacking --meta_learner logreg

   Writes `<meta_learner>_meta.joblib` and `stacking_test_metrics_<meta_learner>.json`.
   Besides the metrics of the two base models and of the meta-learner, the JSON holds the
   plain average of the two models' probabilities (an untrained baseline) and the paired
   comparison of the meta-learner with Mamba, JITrans and that average: McNemar's exact
   test and a bootstrap 95% CI of the accuracy difference.

   | Flag | Meta-learners | Default |
   |---|---|---|
   | `--meta_learner` | | `rf` |
   | `--features` | all (`probs`, `logprobs`; the trees are invariant to it) | `probs` |
   | `--n_estimators` | rf, xgboost, catboost (trees / boosting rounds) | 300 |
   | `--max_depth` | rf, xgboost, catboost | unlimited for rf, 6 for xgboost / catboost |
   | `--class_weight` | all (`balanced`, `none`; `balanced_subsample` rf only) | `balanced` |
   | `--random_state` | all | 42 |
   | `--n_jobs` | rf, xgboost, catboost | -1 |
   | `--min_samples_leaf` | rf | 1 |
   | `--learning_rate` | xgboost, catboost | 0.1 |
   | `--subsample`, `--colsample_bytree` | xgboost | 1.0, 1.0 |
   | `--min_child_weight` | xgboost | 1.0 |
   | `--l2_leaf_reg` | catboost | 3.0 |
   | `--C`, `--max_iter` | logreg (L2, lbfgs) | 1.0, 1000 |

   The meta-learner is trained on the whole val split, so the boosting models run a fixed
   number of rounds, the one chosen by `tune_meta.py`. The defaults (unlimited RF depth,
   500-round boosting) let the trees memorise val, so pass the tuned flags.
   The logistic regression (`logreg`) is multinomial (softmax), with no scaler: with
   `--features logprobs` it learns a weighted product of the two softmaxes, which gives
   better calibrated probabilities than a linear function of the raw probabilities.
   Its `--model-summary` reports parameters and FLOPs instead of the tree statistics.

4. Evaluate it:

       python inference.py --model stacking --meta-model stacking/xgboost_meta.joblib \
           --preds-dir stacking/preds --split test --plot-all --compare \
           --model-summary --benchmark --inference --export-dir ./eval

   Exported files are named after `stacking_<meta_learner>`, so several meta-learners can
   share one export directory. `--rf` is still accepted as an alias of `--meta-model`, and
   `rf_meta.joblib` files written before the meta-learner choice still load. The
   `--features` transform is stored in the joblib and applied by `inference.py`; older
   joblib files are read as `probs`.
