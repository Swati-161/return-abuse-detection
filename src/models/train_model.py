"""
Supervised risk-scoring model for ReturnShield.

Pipeline (brief §5)
───────────────────
  Enhanced feature vector (27 raw features + if_score + reconstruction_error)
    → XGBoost  (scale_pos_weight, NOT SMOTE)
    → CalibratedClassifierCV  (Platt scaling, cv=5)
    → calibrated probability  →  0–100 risk score
    → bands: safe 0–30 / monitor 31–70 / high_risk 71–100

Design decisions locked by the brief
─────────────────────────────────────
  scale_pos_weight, not SMOTE (brief §6)
      SMOTE interpolates new positives in feature space, adding synthetic noise
      to the already-overlapping classes.  scale_pos_weight is mathematically
      equivalent (reweights the gradient) without introducing spurious examples.
      It is also the dominant pattern in production fraud systems and much easier
      to explain in an interview.

  One model (brief §6)
      Stacking XGBoost + LightGBM + meta-learner gives near-zero gain on noisy
      labels.  The two anomaly scores (if_score + reconstruction_error) already
      supply a genuinely different information source; that is the principled
      blend.

  CalibratedClassifierCV, cv=5 (brief §6)
      Raw boosted-tree probabilities are not well-calibrated.  Platt scaling
      (sigmoid) with 5-fold cross-validation uses out-of-fold holdouts so the
      calibrator does not see its own training data.  The resulting probability
      is calibrated to the SIMULATED population — this is documented explicitly
      (brief §8.6) and must be stated when presenting results.

  Evaluation headline: PR-AUC (brief §7)
      The positive (fraud) class is rare (~4%).  PR-AUC is far more informative
      than accuracy or even ROC-AUC under such imbalance.  ROC-AUC is reported
      as a secondary metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
import xgboost as xgb
import joblib
from pathlib import Path

# ── Column contracts ───────────────────────────────────────────────────────────
# These are the three metadata columns that pass through the feature parquets
# but are NEVER fed into any model.

META_COLS: list[str] = ["binary_label", "latent_type", "customer_unique_id"]

# Risk band thresholds (brief §5 architecture)
RISK_BANDS: dict[str, tuple[int, int]] = {
    "safe":      (0,  30),
    "monitor":   (31, 70),
    "high_risk": (71, 100),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return the ordered list of model-input columns (everything except META_COLS).
    This avoids hardcoding the feature list in the runner scripts.
    """
    return [c for c in df.columns if c not in META_COLS]


# ── Model construction ─────────────────────────────────────────────────────────

def build_xgb(scale_pos_weight: float, seed: int = 42) -> xgb.XGBClassifier:
    """
    Construct the base XGBoost classifier.

    Hyperparameter rationale
    ────────────────────────
    max_depth=4
        Shallow trees prevent any single interaction from dominating.  Deep trees
        on noisy synthetic data overfit quickly and produce non-robust SHAP plots.
    min_child_weight=10
        Requires at least 10 weighted samples to split a leaf, preventing splits
        on single rare-class examples.
    subsample=0.8, colsample_bytree=0.8
        Standard stochastic gradient boosting regularisation.
    scale_pos_weight
        Computed from the training set as n_negative / n_positive.  This is the
        recommended XGBoost approach for class-imbalanced binary classification
        (equivalent to oversampling the minority class in the loss function).
    n_estimators=500, learning_rate=0.05
        500 trees × 0.05 step size gives a smooth, well-regularised gradient path.
        With no early stopping (CalibratedClassifierCV does not easily thread
        eval_set through its internal folds), this fixed budget is a safe choice
        at this dataset scale (~77k samples).
    """
    return xgb.XGBClassifier(
        n_estimators       = 500,
        learning_rate      = 0.05,
        max_depth          = 4,
        min_child_weight   = 10,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        scale_pos_weight   = scale_pos_weight,
        objective          = "binary:logistic",
        eval_metric        = "aucpr",
        random_state       = seed,
        n_jobs             = 1,
        tree_method        = "hist",
        verbosity          = 0,
    )


def train_calibrated(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int = 42,
) -> CalibratedClassifierCV:
    """
    Fit a CalibratedClassifierCV wrapping XGBoost on the training data.

    Steps
    ─────
    1. Compute scale_pos_weight from the label distribution.
    2. Build XGBClassifier with that weight.
    3. Wrap in CalibratedClassifierCV(method='sigmoid', cv=5).
       - 'sigmoid' (Platt scaling) is appropriate for XGBoost: the tree model
         already produces decent rank scores; sigmoid corrects the probability
         magnitude without overfitting the calibrator.
       - cv=5 uses out-of-fold predictions so the calibrator never sees its own
         training data — leakage-safe.
    4. Fit on X_train, y_train.

    The returned model's predict_proba() outputs are calibrated probabilities
    averaged across the 5 fold-calibrators.

    Calibration caveat (brief §8.6):
        Calibrated to the simulated population's class distribution (~4% fraud).
        Real-world prevalence may differ; raw probabilities are NOT comparable to
        real-world fraud rates.
    """
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw   = n_neg / max(n_pos, 1)

    print(f"  scale_pos_weight = {n_neg:,} / {n_pos:,} = {spw:.2f}")

    base_xgb  = build_xgb(scale_pos_weight=spw, seed=seed)
    cal_model = CalibratedClassifierCV(base_xgb, method="sigmoid", cv=5)
    cal_model.fit(X_train, y_train)
    return cal_model


# ── Scoring ────────────────────────────────────────────────────────────────────

def prob_to_risk_score(prob: np.ndarray) -> np.ndarray:
    """
    Map calibrated fraud probability → integer 0–100 risk score.
    Linear rescaling; no clipping needed since prob ∈ [0,1].
    """
    return np.clip(np.round(prob * 100), 0, 100).astype(int)


def assign_risk_band(risk_score: np.ndarray | int) -> np.ndarray:
    """
    Map integer risk score → risk band string.
    Thresholds from brief §5: safe 0–30 / monitor 31–70 / high_risk 71–100.
    """
    arr = np.asarray(risk_score)
    out = np.full(arr.shape, "safe", dtype=object)
    out[arr > 30]  = "monitor"
    out[arr > 70]  = "high_risk"
    return out


# ── Evaluation metrics ─────────────────────────────────────────────────────────

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Of the top-k highest-risk customers, what fraction are true fraudsters?
    Actionable framing: "if we investigate the top-k flagged customers,
    how many are genuine abuse cases?"
    """
    if k <= 0 or k > len(y_true):
        return float("nan")
    top_k = np.argsort(y_score)[-k:]
    return float(y_true[top_k].sum() / k)


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Of all true fraudsters, what fraction land in the top-k?
    Operational framing: "what share of the fraud population does a top-k
    review programme capture?"
    """
    total_pos = y_true.sum()
    if total_pos == 0 or k <= 0 or k > len(y_true):
        return float("nan")
    top_k = np.argsort(y_score)[-k:]
    return float(y_true[top_k].sum() / total_pos)


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_list: list[int] | None = None,
) -> dict:
    """
    Full evaluation suite (brief §7).

    Returns
    ───────
    dict with keys:
        pr_auc              — average precision (area under PR curve) [HEADLINE]
        roc_auc             — area under ROC curve
        precision_at_k      — {k: precision} for each k in k_list
        recall_at_k         — {k: recall}    for each k in k_list
        pr_curve            — (precision_arr, recall_arr, thresholds_arr)
        roc_curve           — (fpr_arr, tpr_arr, thresholds_arr)
        n_positive          — total positive labels in y_true
        n_total             — total samples
        prevalence          — n_positive / n_total
    """
    if k_list is None:
        n = len(y_true)
        k_list = [100, 200, 500, int(0.05 * n), int(0.10 * n)]
        k_list = sorted(set(k_list))

    pr_auc  = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    prec_arr, rec_arr, pr_thresh   = precision_recall_curve(y_true, y_prob)
    fpr_arr,  tpr_arr, roc_thresh  = roc_curve(y_true, y_prob)

    return {
        "pr_auc":         pr_auc,
        "roc_auc":        roc_auc,
        "precision_at_k": {k: precision_at_k(y_true, y_prob, k) for k in k_list},
        "recall_at_k":    {k: recall_at_k(y_true, y_prob, k)    for k in k_list},
        "pr_curve":       (prec_arr, rec_arr, pr_thresh),
        "roc_curve":      (fpr_arr, tpr_arr, roc_thresh),
        "n_positive":     int(y_true.sum()),
        "n_total":        len(y_true),
        "prevalence":     float(y_true.mean()),
    }


def print_eval_report(metrics: dict, title: str = "Evaluation Report") -> None:
    """Pretty-print the evaluation dict to stdout."""
    width = 62
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    print(f"  Positive class (fraud): {metrics['n_positive']:,} / "
          f"{metrics['n_total']:,}  "
          f"({metrics['prevalence']:.2%})")
    print()
    print(f"  PR-AUC  (headline):  {metrics['pr_auc']:.4f}   ← brief §7")
    print(f"  ROC-AUC (secondary): {metrics['roc_auc']:.4f}")
    print()
    print(f"  {'K':>6}  {'Precision@K':>13}  {'Recall@K':>10}  "
          f"{'True positives':>15}")
    print("  " + "-" * 50)
    for k in sorted(metrics["precision_at_k"]):
        p = metrics["precision_at_k"][k]
        r = metrics["recall_at_k"][k]
        tp = int(round(p * k)) if not np.isnan(p) else 0
        print(f"  {k:>6}  {p:>13.4f}  {r:>10.4f}  {tp:>15,}")
    print("=" * width)


# ── SHAP helper ────────────────────────────────────────────────────────────────

def extract_xgb_for_shap(cal_model: CalibratedClassifierCV) -> xgb.XGBClassifier:
    """
    Extract the first fold's base XGBoost estimator from a CalibratedClassifierCV.

    CalibratedClassifierCV(cv=5) internally trains 5 XGBoost models (one per fold)
    and fits a sigmoid calibrator on each fold's holdout.  For SHAP, we use the
    first fold's model as a representative.  SHAP values are stable across folds
    because each fold trains on ~80% of the same data distribution.

    The resulting SHAP values reflect the raw XGBoost decision surface before
    calibration.  Feature importances are unaffected by sigmoid rescaling.
    """
    return cal_model.calibrated_classifiers_[0].estimator


# ── Persistence ────────────────────────────────────────────────────────────────

def save_model(model: CalibratedClassifierCV, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  Calibrated model saved → {path}")


def load_model(path: Path) -> CalibratedClassifierCV:
    return joblib.load(path)
