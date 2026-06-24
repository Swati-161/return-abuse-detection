"""
Phase 2 runner — feature engineering + leakage-safe parallel anomaly detection.

Usage
─────
    python run_phase2.py

Prerequisites
─────────────
    python run_phase1.py   # must exist first

Pipeline
────────
  customers_simulated.parquet   (Phase 1 output)
         │
         ├─ build_feature_matrix()        → X_raw  (27 features, unscaled)
         │
         ├─ train_test_split (80/20, stratified on binary_label)
         │   └─ StandardScaler.fit(X_train_raw)
         │        ├─ X_train_scaled
         │        └─ X_test_scaled   (transform only — no fit on test)
         │
         ├─ IsolationForest.fit(X_train_scaled)           ← train only
         │   ├─ if_score_train  =  -decision_function(X_train_scaled)
         │   └─ if_score_test   =  -decision_function(X_test_scaled)
         │      (negated so higher score = more anomalous)
         │
         └─ ReturnAbuseAutoencoder.fit(X_train_scaled)   ← train only
             ├─ recon_error_train  =  per-sample MSE(X_train_scaled)
             └─ recon_error_test   =  per-sample MSE(X_test_scaled)

  Enhanced feature vector fed to XGBoost (Phase 3):
      X_raw  +  if_score  +  reconstruction_error
      (raw, not scaled — XGBoost is tree-based and scale-invariant)

Leakage precautions enforced here (brief §8)
────────────────────────────────────────────
  §8.1  Scaler, IF, and AE all fitted on training split only.
  §8.2  AE trained on full training population (unlabeled bulk majority —
        ~90% normal with mild contamination).  Label knowledge never used
        to select the AE training set.  Documented explicitly here.
  §8.3  Split at the customer level.  Every row is one unique customer, so a
        row-level split is customer-level by construction.
  §8.4  27 features spread across behavioural / temporal / customer dimensions
        so no single variable can carry the bulk of the SHAP attribution.

Outputs
───────
  outputs/features_train.parquet        — enhanced training features + metadata
  outputs/features_test.parquet         — enhanced test features + metadata
  outputs/coupon_abuser_scores.parquet  — IF + AE scores for held-out population
  outputs/models/scaler.joblib
  outputs/models/isolation_forest.joblib
  outputs/models/autoencoder.pt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR, PLOT_DIR, SEED
from src.features.build_features import (
    ALL_FEATURES,
    build_feature_matrix,
    fit_scaler,
    save_scaler,
)
from src.models.autoencoder import (
    train_autoencoder,
    reconstruction_error,
    save_autoencoder,
)

MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
TEST_SIZE     = 0.20   # 80/20 stratified split
IF_N_TREES    = 200    # isolation forest ensemble size
AE_EPOCHS     = 150    # max autoencoder training epochs
AE_BATCH      = 256
AE_LR         = 1e-3
AE_PATIENCE   = 15     # early-stopping patience


# ── Helpers ────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    width = 60
    bar = "─" * max(0, width - len(title) - 1)
    print(f"\n── {title} {bar}")


def _score_stats_by_label(
    scores: np.ndarray,
    labels: np.ndarray,
    score_name: str,
    indent: str = "  ",
) -> None:
    """Print mean / p75 anomaly scores split by binary_label."""
    print(f"{indent}{score_name} (higher = more anomalous):")
    for lab, name in [(0, "non-fraud"), (1, "fraud   ")]:
        mask = labels == lab
        if mask.sum() == 0:
            continue
        m   = scores[mask].mean()
        p75 = np.percentile(scores[mask], 75)
        print(f"{indent}  {name}: n={mask.sum():>6,}  mean={m:>8.4f}  p75={p75:>8.4f}")


def _score_stats_by_type(
    df: pd.DataFrame,
    score_col: str,
    type_col: str = "latent_type",
    indent: str = "  ",
) -> None:
    """Print mean / p75 per latent archetype — the key per-type sanity check."""
    order = ["normal", "impulse", "serial", "fraud"]
    print(f"{indent}{score_col} by latent type:")
    print(f"{indent}  {'type':<10} {'n':>6}  {'mean':>8}  {'p75':>8}")
    print(f"{indent}  " + "-" * 38)
    for t in order:
        sub = df.loc[df[type_col] == t, score_col]
        if len(sub) == 0:
            continue
        print(f"{indent}  {t:<10} {len(sub):>6}  "
              f"{sub.mean():>8.4f}  {np.percentile(sub, 75):>8.4f}")


# ── Visualisation ──────────────────────────────────────────────────────────────

def _plot_anomaly_score_distributions(
    test_df: pd.DataFrame,
    score_cols: list[str],
    labels: list[str],
) -> None:
    """
    KDE of each anomaly score per latent type on the test set.
    Saved to outputs/plots/phase2_anomaly_scores.png.
    """
    colors = {
        "normal": "#4C72B0", "impulse": "#55A868",
        "serial": "#C44E52", "fraud":   "#8172B2",
    }
    order = ["normal", "impulse", "serial", "fraud"]

    fig, axes = plt.subplots(1, len(score_cols), figsize=(7 * len(score_cols), 5))
    if len(score_cols) == 1:
        axes = [axes]

    for ax, col, label in zip(axes, score_cols, labels):
        for t in order:
            sub = test_df.loc[test_df["latent_type"] == t, col]
            if len(sub) < 5:
                continue
            sub.plot.kde(ax=ax, label=t, color=colors[t], linewidth=2)
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel("density", fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.legend(title="latent type")

    fig.suptitle(
        "Anomaly score distributions per archetype (test set)\n"
        "Fraud should shift right — but overlap with serial is expected",
        fontsize=11,
    )
    fig.tight_layout()
    path = PLOT_DIR / "phase2_anomaly_scores.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path}")


def _plot_val_loss(val_history: list[float]) -> None:
    """Autoencoder validation MSE curve — confirms convergence."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(val_history, color="#4C72B0", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation MSE", fontsize=11)
    ax.set_title("Autoencoder training — validation loss curve", fontsize=11)
    fig.tight_layout()
    path = PLOT_DIR / "phase2_ae_val_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── 0. Load Phase 1 outputs ───────────────────────────────────────────────
    _section("Loading Phase 1 outputs")
    sim_path    = OUTPUT_DIR / "customers_simulated.parquet"
    coupon_path = OUTPUT_DIR / "coupon_abusers.parquet"

    if not sim_path.exists():
        sys.exit(
            f"\nERROR: {sim_path} not found.\n"
            "Run `python run_phase1.py` first, then re-run this script.\n"
        )

    df = pd.read_parquet(sim_path)
    print(f"Loaded {len(df):,} simulated customers.")

    ca_df: pd.DataFrame | None = None
    if coupon_path.exists():
        ca_df = pd.read_parquet(coupon_path)
        print(f"Loaded {len(ca_df):,} held-out coupon abusers.")
    else:
        print("  coupon_abusers.parquet not found — held-out scoring will be skipped.")

    print("\nArchetype distribution:")
    for t, cnt in df["latent_type"].value_counts().items():
        print(f"  {t:>10s}: {cnt:>7,}  ({100 * cnt / len(df):.1f}%)")
    print(f"\nFraud prevalence (binary_label=1): {df['binary_label'].mean():.3%}")

    # ── 1. Feature engineering ────────────────────────────────────────────────
    _section("Feature engineering")
    X_raw, feat_names = build_feature_matrix(df)
    n_features = len(feat_names)
    print(f"Feature matrix: {X_raw.shape}  ({n_features} features)")
    print(f"Feature groups:")
    print(f"  Behavioural (7): return_ratio, high_value_return_rate, coupon_usage_rate ...")
    print(f"  Temporal    (5): returns_last_7/30/90, order_frequency, days_since_last_order")
    print(f"  Customer    (5): account_age_days, avg_order_value, CLV ...")
    print(f"  Binary flags(6): high_value_return, coupon_heavy, short_account ...")
    print(f"  Derived     (4): return_burst_rate, return_acceleration, clv_per_order, "
          "return_7d_fraction")

    y       = df["binary_label"].values
    cust_id = df["customer_unique_id"].astype(str).to_numpy()
    lat_typ = df["latent_type"].astype(str).to_numpy()

    y = np.asarray(y)

    X_raw = np.asarray(X_raw)

    # ── 2. Stratified train / test split at customer level ────────────────────
    #
    # Brief §8.3: one customer must never appear in both train and test.
    # Every row here is already a unique customer, so a row-level stratified
    # split is a customer-level split by construction.
    # Stratify on binary_label to keep the 3% fraud class in both splits.
    _section("Train / test split  (80/20, stratified)")
    (
        X_train_raw, X_test_raw,
        y_train,     y_test,
        id_train,    id_test,
        lt_train,    lt_test,
    ) = train_test_split(
        X_raw, y, cust_id, lat_typ,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    print(f"Train: {len(X_train_raw):>7,} customers  ({y_train.mean():.3%} fraud)")
    print(f"Test:  {len(X_test_raw):>7,} customers  ({y_test.mean():.3%} fraud)")

    # ── 3. Fit StandardScaler on train only  ──────────────────────────────────
    #
    # Brief §8.1: fitting on the full dataset before splitting inflates scores.
    # The scaler is a transformer with statistics computed from training data;
    # applying it to test data is not leakage.
    _section("StandardScaler  (fit on train, apply to test)")
    scaler = fit_scaler(X_train_raw)
    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)
    save_scaler(scaler, MODEL_DIR / "scaler.joblib")

    # ── 4. Isolation Forest  (fit on train only) ──────────────────────────────
    #
    # contamination='auto' — only affects the binary predict() output, which we
    # never use.  We use the raw decision_function scores throughout.
    # n_jobs=-1 uses all CPU cores for the ensemble.
    _section("Isolation Forest  (fit on X_train_scaled)")
    print(f"  n_estimators={IF_N_TREES}, max_samples='auto', contamination='auto'")
    iso = IsolationForest(
        n_estimators  = IF_N_TREES,
        max_samples   = "auto",
        contamination = "auto",
        random_state  = SEED,
        n_jobs        = -1,
    )
    iso.fit(X_train_scaled)
    joblib.dump(iso, MODEL_DIR / "isolation_forest.joblib")
    print(f"  Saved → {MODEL_DIR / 'isolation_forest.joblib'}")

    # Negate decision_function: sklearn returns positive=normal, negative=outlier.
    # We flip sign so higher if_score = more anomalous (consistent with AE convention).
    if_train = -iso.decision_function(X_train_scaled)
    if_test  = -iso.decision_function(X_test_scaled)

    print("\n  Isolation Forest score sanity check:")
    _score_stats_by_label(if_train, y_train, "IF score [train]", indent="  ")
    _score_stats_by_label(if_test,  y_test,  "IF score [test]",  indent="  ")

    # ── 5. Autoencoder  (fit on train only, unlabeled bulk majority) ──────────
    #
    # Brief §8.2 — two defensible options:
    #   (A) Train on the FULL training split (~90% normal, mild contamination).
    #       Production-defensible: labels are not used to select the training set.
    #   (B) Train on known-"normal" customers only (uses label knowledge).
    #
    # We use option (A).  The ~10% contamination from serial/fraud customers
    # slightly raises the reconstruction floor for those patterns, but in a real
    # deployment this is the only honest setup.  Document this explicitly here
    # so the choice is auditable.
    _section("Autoencoder  (fit on X_train_scaled — unlabeled bulk majority)")
    print(f"  Architecture: {n_features} → 16 → 8 → 16 → {n_features}")
    print(f"  Training on full training split ({len(X_train_scaled):,} customers)")
    print(f"  Rationale: ~90% are normal/impulse; using label knowledge to filter")
    print(f"  would be production-indefensible (brief §8.2 option A).")

    ae_model, val_history = train_autoencoder(
        X_train_scaled,
        n_epochs   = AE_EPOCHS,
        batch_size = AE_BATCH,
        lr         = AE_LR,
        patience   = AE_PATIENCE,
        seed       = SEED,
        verbose    = True,
    )
    save_autoencoder(ae_model, MODEL_DIR / "autoencoder.pt")
    _plot_val_loss(val_history)

    ae_train = reconstruction_error(ae_model, X_train_scaled)
    ae_test  = reconstruction_error(ae_model, X_test_scaled)

    print("\n  Autoencoder reconstruction error sanity check:")
    _score_stats_by_label(ae_train, y_train, "AE recon error [train]", indent="  ")
    _score_stats_by_label(ae_test,  y_test,  "AE recon error [test]",  indent="  ")

    # ── 6. Build enhanced feature vectors ─────────────────────────────────────
    #
    # XGBoost is tree-based and scale-invariant → use raw (unscaled) features.
    # Anomaly scores are appended as two extra model inputs.
    # The metadata columns (latent_type, binary_label, customer_unique_id) pass
    # through for evaluation; they are NOT model inputs.
    _section("Building enhanced feature vectors for Phase 3")

    def _assemble(
        X: np.ndarray,
        y: np.ndarray,
        cust_ids: np.ndarray,
        lat_types: np.ndarray,
        if_scores: np.ndarray,
        ae_scores: np.ndarray,
    ) -> pd.DataFrame:
        feat_df = pd.DataFrame(X, columns=feat_names)
        feat_df["if_score"]             = if_scores
        feat_df["reconstruction_error"] = ae_scores
        feat_df["binary_label"]         = y
        feat_df["latent_type"]          = lat_types
        feat_df["customer_unique_id"]   = cust_ids
        return feat_df

    train_df = _assemble(X_train_raw, y_train, id_train, lt_train, if_train, ae_train)
    test_df  = _assemble(X_test_raw,  y_test,  id_test,  lt_test,  if_test,  ae_test)

    model_input_cols = feat_names + ["if_score", "reconstruction_error"]
    print(f"  Train enhanced shape: {train_df.shape}")
    print(f"  Test  enhanced shape: {test_df.shape}")
    print(f"  Model input columns ({len(model_input_cols)}): "
          f"[{', '.join(model_input_cols[:4])}, ..., if_score, reconstruction_error]")

    # ── 7. Score held-out coupon abusers ──────────────────────────────────────
    #
    # Coupon abusers were NEVER in training — not in feature engineering,
    # not in the scaler fit, not in IF/AE training.  We apply the train-fitted
    # scaler and the train-fitted models to their features, exactly as we would
    # at inference time in production.
    _section("Scoring held-out coupon abusers (never in training)")
    ca_scores_df: pd.DataFrame | None = None

    if ca_df is not None:
        X_ca_raw, _ = build_feature_matrix(ca_df)
        X_ca_scaled = scaler.transform(X_ca_raw)        # train scaler only

        if_ca = -iso.decision_function(X_ca_scaled)    # train IF only
        ae_ca = reconstruction_error(ae_model, X_ca_scaled)   # train AE only

        print(f"\n  Coupon abuser IF score:   "
              f"mean={if_ca.mean():.4f}  "
              f"p75={np.percentile(if_ca, 75):.4f}")
        print(f"  Coupon abuser AE error:   "
              f"mean={ae_ca.mean():.4f}  "
              f"p75={np.percentile(ae_ca, 75):.4f}")

        print(f"\n  Comparison — test-set true fraud:")
        fraud_mask = y_test == 1
        print(f"    IF score:   mean={if_test[fraud_mask].mean():.4f}  "
              f"p75={np.percentile(if_test[fraud_mask], 75):.4f}")
        print(f"    AE error:   mean={ae_test[fraud_mask].mean():.4f}  "
              f"p75={np.percentile(ae_test[fraud_mask], 75):.4f}")

        print(f"\n  Comparison — test-set normal buyers:")
        normal_mask = (test_df["latent_type"] == "normal").values
        print(f"    IF score:   mean={if_test[normal_mask].mean():.4f}  "
              f"p75={np.percentile(if_test[normal_mask], 75):.4f}")
        print(f"    AE error:   mean={ae_test[normal_mask].mean():.4f}  "
              f"p75={np.percentile(ae_test[normal_mask], 75):.4f}")

        print("\n  Interpretation:")
        print("  Coupon abusers have DIFFERENT signature from fraud — their")
        print("  return_rate is moderate and account_age is long.  XGBoost")
        print("  (trained to flag high-return-rate fraud) should miss many of them,")
        print("  while IF/AE may still flag them as 'unusual'.  Confirmed in Phase 6.")

        ca_scores_df = pd.DataFrame({
            "latent_type":         "coupon_abuser",
            "if_score":            if_ca,
            "reconstruction_error": ae_ca,
            "binary_label":        0,
        })
    else:
        print("  Skipped (coupon_abusers.parquet not found).")

    # ── 8. Per-type anomaly score breakdown (test set) ────────────────────────
    _section("Anomaly score summary by latent type  (test set)")
    print()
    _score_stats_by_type(test_df, "if_score")
    print()
    _score_stats_by_type(test_df, "reconstruction_error")

    # ── 9. Plots ──────────────────────────────────────────────────────────────
    _section("Generating anomaly score distribution plots")
    _plot_anomaly_score_distributions(
        test_df,
        score_cols=["if_score", "reconstruction_error"],
        labels=[
            "Isolation Forest score\n(higher = more anomalous)",
            "AE reconstruction error\n(higher = more anomalous)",
        ],
    )

    # ── 10. Save outputs ──────────────────────────────────────────────────────
    _section("Saving outputs")
    train_path = OUTPUT_DIR / "features_train.parquet"
    test_path  = OUTPUT_DIR / "features_test.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path,   index=False)
    print(f"  {train_path}")
    print(f"  {test_path}")

    if ca_scores_df is not None:
        ca_path = OUTPUT_DIR / "coupon_abuser_scores.parquet"
        ca_scores_df.to_parquet(ca_path, index=False)
        print(f"  {ca_path}")

    # ── 11. Summary ───────────────────────────────────────────────────────────
    _section("Phase 2 complete")
    print(f"""
  What was built:
    ✓ Feature matrix with {n_features} engineered features
      (behavioural / temporal / customer / binary-flags / derived)
    ✓ StandardScaler  →  outputs/models/scaler.joblib
    ✓ Isolation Forest ({IF_N_TREES} trees)  →  outputs/models/isolation_forest.joblib
    ✓ Autoencoder ({n_features}→16→8→16→{n_features})  →  outputs/models/autoencoder.pt
    ✓ Enhanced feature vectors (raw features + if_score + reconstruction_error)
        Train: outputs/features_train.parquet
        Test:  outputs/features_test.parquet
    ✓ Held-out coupon abuser anomaly scores (for Phase 6 archetype experiment)

  Leakage status: CLEAN
    - Scaler, IF, AE all fitted on training split only.
    - AE trained on full training population (unlabeled bulk majority — brief §8.2A).
    - Coupon abusers never touched any fitting step.

  Next step: python run_phase3.py   (XGBoost + calibration + SHAP)
""")


if __name__ == "__main__":
    main()
