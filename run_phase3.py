"""
Phase 3 runner — supervised risk scoring, calibration, and SHAP explainability.

Usage
─────
    python run_phase3.py

Prerequisites
─────────────
    python run_phase2.py   # features_train.parquet + features_test.parquet

Pipeline
────────
  features_train.parquet  +  features_test.parquet   (Phase 2 outputs)
         │
         ├─ Separate model inputs (27 features + if_score + reconstruction_error)
         │  from metadata (binary_label / latent_type / customer_unique_id)
         │
         ├─ XGBoost(scale_pos_weight)              ← class weights, NOT SMOTE
         │      wrapped in CalibratedClassifierCV  ← sigmoid, cv=5
         │      fitted on X_train  only            ← train split boundary respected
         │
         ├─ Evaluation on X_test (never seen during training or calibration)
         │      PR-AUC (headline)  ·  ROC-AUC  ·  Precision@K  ·  Recall@K
         │
         ├─ SHAP (TreeExplainer on first-fold base XGBoost)
         │      summary plot (beeswarm)  ·  bar plot
         │
         ├─ Diagnostic plots
         │      PR curve  ·  ROC curve  ·  risk-score distribution by archetype
         │
         └─ Manual top-20 review (highest-risk test customers)

Leakage status: CLEAN
    - Model fitted on training split columns only.
    - Test set scores computed after fit, never during.
    - Calibration uses cv=5 out-of-fold holdouts within the training split.
    - Coupon abuser scoring uses the model as-is (no re-fitting).

Outputs
───────
    outputs/models/calibrated_xgboost.joblib
    outputs/plots/shap_summary.png          (beeswarm)
    outputs/plots/shap_bar.png              (mean |SHAP| ranking)
    outputs/plots/phase3_pr_curve.png
    outputs/plots/phase3_roc_curve.png
    outputs/plots/phase3_score_distribution.png
    outputs/test_scored.parquet             (test set + risk_score + risk_band)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.config import OUTPUT_DIR, PLOT_DIR, SEED
from src.features.build_features import build_feature_matrix, load_scaler
from src.models.autoencoder import reconstruction_error as ae_reconstruction_error
from src.models.autoencoder import load_autoencoder
from src.models.train_model import (
    META_COLS,
    get_feature_cols,
    train_calibrated,
    prob_to_risk_score,
    assign_risk_band,
    evaluate,
    print_eval_report,
    extract_xgb_for_shap,
    save_model,
)

MODEL_DIR = OUTPUT_DIR / "models"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    bar = "─" * max(0, 59 - len(title))
    print(f"\n── {title} {bar}")


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_pr_curve(metrics: dict, out_path: Path) -> None:
    prec, rec, _ = metrics["pr_curve"]
    pr_auc       = metrics["pr_auc"]
    baseline     = metrics["prevalence"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, color="#C44E52", linewidth=2,
            label=f"XGBoost calibrated  (PR-AUC = {pr_auc:.4f})")
    ax.axhline(baseline, linestyle="--", color="#888", linewidth=1,
               label=f"Random baseline  ({baseline:.3f})")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve\n"
                 "(PR-AUC is the headline metric — brief §7)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


def _plot_roc_curve(metrics: dict, out_path: Path) -> None:
    fpr, tpr, _ = metrics["roc_curve"]
    roc_auc     = metrics["roc_auc"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4C72B0", linewidth=2,
            label=f"XGBoost calibrated  (ROC-AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888", linewidth=1,
            label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve (secondary metric — brief §7)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


def _plot_score_distribution(scored_df: pd.DataFrame, out_path: Path) -> None:
    """
    KDE of 0–100 risk score per latent type on the test set.
    Fraud should shift right; the distributions MUST overlap (brief §2).
    """
    colors = {
        "normal": "#4C72B0", "impulse": "#55A868",
        "serial": "#C44E52", "fraud":   "#8172B2",
    }
    order = ["normal", "impulse", "serial", "fraud"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for t in order:
        sub = scored_df.loc[scored_df["latent_type"] == t, "risk_score"].astype(float)
        if len(sub) < 10:
            continue
        sub.plot.kde(ax=ax, label=t, color=colors[t], linewidth=2, bw_method=0.15)

    ax.set_xlabel("Risk score (0–100)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Risk score distribution by latent archetype (test set)\n"
        "Overlap between serial and fraud is expected and desired — brief §2",
        fontsize=11,
    )
    ax.axvline(31, color="#888", linestyle="--", linewidth=1, alpha=0.7,
               label="monitor threshold (31)")
    ax.axvline(71, color="#333", linestyle="--", linewidth=1, alpha=0.7,
               label="high-risk threshold (71)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


def _plot_shap_summary(shap_values: np.ndarray,
                       X_shap: np.ndarray,
                       feature_names: list[str],
                       out_path_beeswarm: Path,
                       out_path_bar: Path,
                       max_display: int = 20) -> None:
    """
    Generate and save:
      1. SHAP beeswarm summary plot  (each dot = one test customer)
      2. SHAP bar plot               (mean |SHAP value| per feature)

    Both use the first-fold XGBoost estimator extracted from the calibrated model.
    """
    shap_df = pd.DataFrame(X_shap, columns=feature_names)

    # ── Beeswarm ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, shap_df,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    ax = plt.gca()
    ax.set_title(
        "SHAP Feature Impact — ReturnShield\n"
        "(each dot = one test customer; colour = feature value)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path_beeswarm, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP beeswarm saved → {out_path_beeswarm}")

    # ── Bar (mean |SHAP|) ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(
        shap_values, shap_df,
        max_display=max_display,
        plot_type="bar",
        show=False,
        plot_size=None,
    )
    ax = plt.gca()
    ax.set_title(
        "SHAP Mean Absolute Feature Importance\n"
        "(brief §8.4: no single feature should dominate)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path_bar, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP bar saved     → {out_path_bar}")


def _print_top_k_customers(
    scored_df: pd.DataFrame,
    feature_cols: list[str],
    k: int = 20,
) -> None:
    """
    Print the top-k highest-risk customers from the test set.
    Shows risk score, true label, latent type, and key features.
    Brief §7: "Manual review of the top 20 highest-risk customers —
    do they look obviously suspicious?"
    """
    top = (
        scored_df
        .sort_values("risk_score", ascending=False)
        .head(k)
        [["customer_unique_id", "risk_score", "risk_band",
          "binary_label", "latent_type",
          "return_ratio", "coupon_usage_rate",
          "high_value_return_rate", "account_age_days",
          "if_score", "reconstruction_error"]]
        .reset_index(drop=True)
    )
    top.index += 1
    print(top.to_string(
        float_format=lambda x: f"{x:.3f}",
        index=True,
    ))


# ── Held-out coupon abuser scoring ─────────────────────────────────────────────

def _score_coupon_abusers(cal_model, feature_cols: list[str]) -> None:
    """
    Score the held-out coupon-abuser population with the trained XGBoost model.

    Brief §7: "Show that XGBoost misses many of them while Isolation Forest
    and the autoencoder flag a meaningful share."

    Steps:
      1. Load raw coupon_abusers.parquet (original simulation rows).
      2. Build raw feature matrix (27 columns).
      3. Apply the Phase-2 train-fitted scaler + IF + AE to get anomaly scores.
         (These were already computed in Phase 2; we recompute here to keep
         this function self-contained and avoid stale-score drift.)
      4. Assemble the full 29-column input vector and call predict_proba().
      5. Print the XGBoost risk-score distribution for the coupon abusers.

    Full archetype experiment (XGBoost vs IF vs AE recovery rates) is
    reported in run_phase6.py.
    """
    ca_raw_path = OUTPUT_DIR / "coupon_abusers.parquet"
    if not ca_raw_path.exists():
        print("  coupon_abusers.parquet not found — skipping.")
        return

    ca_raw = pd.read_parquet(ca_raw_path)
    print(f"  Coupon abuser population: {len(ca_raw):,}")

    # Build raw features (27 columns) — same pipeline as Phase 2
    X_ca_raw, raw_feat_names = build_feature_matrix(ca_raw)

    # Apply Phase-2 train-fitted artifacts (scaler, IF, AE)
    scaler   = load_scaler(MODEL_DIR / "scaler.joblib")
    iso      = joblib.load(MODEL_DIR / "isolation_forest.joblib")
    ae_model = load_autoencoder(MODEL_DIR / "autoencoder.pt")

    X_ca_scaled = scaler.transform(X_ca_raw)
    if_ca       = -iso.decision_function(X_ca_scaled)
    ae_ca       = ae_reconstruction_error(ae_model, X_ca_scaled)

    # Assemble 29-column input: raw features + if_score + reconstruction_error
    # np.column_stack treats 1-D arrays as columns, giving shape (n, 29)
    n_raw_feats = len(raw_feat_names)
    if len(feature_cols) == n_raw_feats + 2:
        # Expected path: feature_cols = 27 raw + if_score + reconstruction_error
        ca_X = np.column_stack([X_ca_raw, if_ca, ae_ca])
    else:
        # Fallback if anomaly scores are absent from the feature vector
        ca_X = X_ca_raw

    ca_prob = cal_model.predict_proba(ca_X)[:, 1]
    ca_risk = prob_to_risk_score(ca_prob)
    ca_band = assign_risk_band(ca_risk)

    safe_pct      = (ca_risk <= 30).mean()
    monitor_pct   = ((ca_risk >= 31) & (ca_risk <= 70)).mean()
    high_risk_pct = (ca_risk > 70).mean()

    print(f"\n  XGBoost risk score distribution for coupon abusers:")
    print(f"    safe       (0–30):  {safe_pct:.1%}")
    print(f"    monitor   (31–70):  {monitor_pct:.1%}")
    print(f"    high_risk (71–100): {high_risk_pct:.1%}")
    print(f"    mean risk score:    {ca_risk.mean():.1f}")
    print()
    print("  Interpretation:")
    print("  XGBoost learned the fraud signature: short tenure + high return rate")
    print("  + high-value items.  Coupon abusers show LONG tenure + MODERATE returns")
    print("  → XGBoost should score most of them low (safe/monitor).")
    print("  IF and AE may still flag them as 'unusual' via reconstruction error.")
    print("  Full cross-model comparison → run_phase6.py")

    scored_ca = pd.DataFrame({
        "latent_type":          "coupon_abuser",
        "risk_score":           ca_risk,
        "risk_band":            ca_band,
        "xgb_fraud_prob":       ca_prob,
        "if_score":             if_ca,
        "reconstruction_error": ae_ca,
    })
    ca_out = OUTPUT_DIR / "coupon_abuser_scored.parquet"
    scored_ca.to_parquet(ca_out, index=False)
    print(f"\n  Saved → {ca_out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── 0. Load Phase 2 outputs ───────────────────────────────────────────────
    _section("Loading Phase 2 enhanced feature parquets")
    train_path = OUTPUT_DIR / "features_train.parquet"
    test_path  = OUTPUT_DIR / "features_test.parquet"

    for p in (train_path, test_path):
        if not p.exists():
            sys.exit(
                f"\nERROR: {p} not found.\n"
                "Run `python run_phase2.py` first.\n"
            )

    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path)
    print(f"Train: {train_df.shape}  |  fraud={train_df.binary_label.mean():.3%}")
    print(f"Test:  {test_df.shape}   |  fraud={test_df.binary_label.mean():.3%}")

    # ── 1. Separate model inputs from metadata ────────────────────────────────
    #
    # get_feature_cols() returns every column that is NOT in META_COLS.
    # This is the 29-column input vector (27 features + if_score + recon_error).
    # binary_label, latent_type, customer_unique_id are metadata — NEVER
    # passed to the model.
    _section("Preparing model inputs")
    feature_cols = get_feature_cols(train_df)
    print(f"Model input features ({len(feature_cols)}): "
          f"[{', '.join(feature_cols[:4])}, ..., "
          f"if_score, reconstruction_error]")

    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df["binary_label"].values.astype(int)

    X_test  = test_df[feature_cols].values.astype(np.float64)
    y_test  = test_df["binary_label"].values.astype(int)
    lt_test = test_df["latent_type"].values

    print(f"\nTrain: X={X_train.shape}  y={y_train.shape}  "
          f"fraud={y_train.sum():,} ({y_train.mean():.3%})")
    print(f"Test:  X={X_test.shape}  y={y_test.shape}  "
          f"fraud={y_test.sum():,} ({y_test.mean():.3%})")

    # ── 2. Train XGBoost + CalibratedClassifierCV ─────────────────────────────
    #
    # Fitted on X_train, y_train only.
    # CalibratedClassifierCV(cv=5) internally creates 5 XGBoost+calibrator pairs
    # using out-of-fold holdouts — the calibrator never sees its own training fold.
    _section("Training: XGBoost + CalibratedClassifierCV")
    print("  XGBoost hyperparameters: n_estimators=500, lr=0.05, max_depth=4")
    print("  Calibration: sigmoid (Platt), cv=5  — calibrated to simulated population")
    print("  Training 5-fold calibration (this may take 2-5 minutes) ...")

    cal_model = train_calibrated(X_train, y_train, seed=SEED)
    print("  Training complete.")

    # ── 3. Score the test set ─────────────────────────────────────────────────
    _section("Scoring test set → 0–100 risk scores")
    y_prob_test = cal_model.predict_proba(X_test)[:, 1]
    risk_scores = prob_to_risk_score(y_prob_test)
    risk_bands  = assign_risk_band(risk_scores)

    scored_test = test_df.copy()
    scored_test["risk_score"] = risk_scores
    scored_test["risk_band"]  = risk_bands
    scored_test["fraud_prob"] = y_prob_test

    print(f"\n  Risk band distribution (test set):")
    for band in ["safe", "monitor", "high_risk"]:
        cnt = (risk_bands == band).sum()
        pct = cnt / len(risk_bands)
        # Of customers in this band, how many are true fraud?
        band_fraud = y_test[risk_bands == band].mean() if cnt > 0 else 0.0
        print(f"    {band:<10}: {cnt:>5,} ({pct:.1%})  "
              f"| fraud rate in band: {band_fraud:.1%}")

    # ── 4. Evaluate on test set ───────────────────────────────────────────────
    _section("Evaluation (test set, against latent labels)")
    n_test = len(y_test)
    metrics = evaluate(
        y_test,
        y_prob_test,
        k_list=[100, 200, 500, int(0.05 * n_test), int(0.10 * n_test)],
    )
    print_eval_report(metrics, title="Phase 3 — Test Set Evaluation")

    # Sanity check: brief §2 "if your model gets ~100% accuracy, data is circular"
    from sklearn.metrics import accuracy_score
    y_pred = (y_prob_test >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy at 0.5 threshold: {acc:.4f}")
    if acc > 0.98:
        print("  ⚠  WARNING: accuracy > 98% — check for data circularity (brief §2).")
    else:
        print("  ✓  Accuracy < 98% — genuine overlap in the data confirmed.")

    # ── 5. Score breakdown by latent type ─────────────────────────────────────
    _section("Risk score by latent archetype (test set)")
    order = ["normal", "impulse", "serial", "fraud"]
    print(f"  {'type':<10}  {'n':>6}  {'mean score':>11}  "
          f"{'p75':>6}  {'high-risk %':>12}")
    print("  " + "-" * 52)
    for t in order:
        mask = lt_test == t
        if mask.sum() == 0:
            continue
        rs = risk_scores[mask]
        hr = (rs > 70).mean()
        print(f"  {t:<10}  {mask.sum():>6}  {rs.mean():>11.2f}  "
              f"{np.percentile(rs, 75):>6.1f}  {hr:>12.1%}")

    # ── 6. Manual top-20 review ───────────────────────────────────────────────
    _section("Manual review — top 20 highest-risk customers (brief §7)")
    print("  (risk_score / true label / archetype / key features)\n")
    _print_top_k_customers(scored_test, feature_cols, k=20)

    # ── 7. SHAP explainability ────────────────────────────────────────────────
    _section("SHAP explainability")
    xgb_for_shap = extract_xgb_for_shap(cal_model)
    print(f"  Using fold-0 XGBoost estimator from CalibratedClassifierCV.")

    # Limit to 5000 test samples for SHAP speed; shuffle to avoid selection bias
    rng       = np.random.default_rng(SEED)
    n_shap    = min(5_000, len(X_test))
    shap_idx  = rng.choice(len(X_test), n_shap, replace=False)
    X_shap    = X_test[shap_idx]
    print(f"  Computing SHAP TreeExplainer on {n_shap:,} test samples ...")

    explainer   = shap.TreeExplainer(xgb_for_shap)
    shap_values = explainer.shap_values(X_shap)

    # For binary XGBoost the output is a single 2-D array (n_samples, n_features)
    if isinstance(shap_values, list):
        # Some SHAP versions return [neg_class, pos_class]
        shap_values = shap_values[1]

    _plot_shap_summary(
        shap_values, X_shap, feature_cols,
        out_path_beeswarm = PLOT_DIR / "shap_summary.png",
        out_path_bar      = PLOT_DIR / "shap_bar.png",
        max_display       = 20,
    )

    # SHAP dominance check — brief §8.4
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_feature = feature_cols[np.argmax(mean_abs)]
    top_share   = mean_abs.max() / mean_abs.sum()
    print(f"\n  Top SHAP feature: '{top_feature}'  "
          f"({top_share:.1%} of total mean |SHAP|)")
    if top_share > 0.60:
        print("  ⚠  WARNING: one feature carries >60% of SHAP attribution.")
        print("  ⚠  This may indicate data circularity — review brief §8.4.")
    else:
        print("  ✓  No single feature dominates SHAP — signal spread confirmed.")

    # ── 8. Diagnostic plots ───────────────────────────────────────────────────
    _section("Generating diagnostic plots")
    _plot_pr_curve(metrics,  PLOT_DIR / "phase3_pr_curve.png")
    _plot_roc_curve(metrics, PLOT_DIR / "phase3_roc_curve.png")
    _plot_score_distribution(scored_test, PLOT_DIR / "phase3_score_distribution.png")

    # ── 9. Score held-out coupon abusers ──────────────────────────────────────
    _section("Scoring held-out coupon abusers with trained model")
    _score_coupon_abusers(cal_model, feature_cols)

    # ── 10. Save model and scored test set ────────────────────────────────────
    _section("Saving outputs")
    save_model(cal_model, MODEL_DIR / "calibrated_xgboost.joblib")

    scored_path = OUTPUT_DIR / "test_scored.parquet"
    scored_test.to_parquet(scored_path, index=False)
    print(f"  Scored test set    → {scored_path}")

    # ── 11. Summary ───────────────────────────────────────────────────────────
    _section("Phase 3 complete")
    print(f"""
  What was built:
    ✓ XGBoost (scale_pos_weight={int(round((1 - y_train.mean()) / y_train.mean()))}) +
      CalibratedClassifierCV(sigmoid, cv=5)
    ✓ 0–100 risk scores + band labels (safe/monitor/high_risk)
    ✓ PR-AUC={metrics['pr_auc']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}
    ✓ SHAP beeswarm + bar plots  →  outputs/plots/shap_summary.png
    ✓ Calibrated model           →  outputs/models/calibrated_xgboost.joblib
    ✓ Scored test set            →  outputs/test_scored.parquet

  Calibration note (brief §8.6):
    Scores calibrated to the simulated population (~4% fraud).
    Real-world fraud prevalence will differ; do not interpret 0–100 scores
    as absolute probabilities outside this simulation context.

  Next step: python run_phase4.py   (SHAP waterfall + policy simulator)
""")


if __name__ == "__main__":
    main()
