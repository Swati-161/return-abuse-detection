"""
Feature engineering pipeline for ReturnShield.

Responsibilities
────────────────
  1. Select and group the ~27 engineered features from the simulated customer
     dataset into behavioural / temporal / customer / binary-flag buckets.
  2. Compute four derived interaction features (return-burst rates, acceleration,
     CLV-per-order) that expose nonlinear joint signals no single raw column
     captures alone.
  3. Expose build_feature_matrix() — returns (X_raw, feature_names) where
     X_raw is unscaled and may be passed directly to XGBoost (tree-based,
     scale-invariant) or fed into a StandardScaler before Isolation Forest / AE.
  4. Expose fit_scaler() / save_scaler() / load_scaler() so the scaler lifecycle
     is managed explicitly by the caller.

Leakage guard
─────────────
  latent_type, binary_label, and customer_unique_id are NEVER included in the
  feature matrix returned here.  Any scaler MUST be fitted on training data only
  and applied to test / holdout data — this module does not enforce the split,
  the caller does.

No PCA
──────
  With ~27 features, PCA destroys interpretability and is incompatible with
  per-customer SHAP waterfalls (brief §4).  All features are passed as-is.

Signal spread (brief §8.4)
──────────────────────────
  The brief warns against letting one variable dominate SHAP, which would
  suggest the label was defined by that variable.  The feature set deliberately
  spreads signal across behavioural, temporal, and customer-lifetime dimensions
  so no single column carries the bulk of the attribution.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


# ── Feature groups ─────────────────────────────────────────────────────────────
#
# Listed in semantically cohesive order — this order is preserved in the
# output array and used by SHAP for grouping.

BEHAVIORAL_FEATURES: list[str] = [
    "total_orders",           # purchase volume
    "return_ratio",           # fraction of orders returned (key signal, but noisy)
    "high_value_return_rate", # fraction of returns that are high-ticket
    "coupon_usage_rate",      # coupon dependency
    "avg_days_to_return",     # days between delivery and return (fraudsters return fast)
    "payment_boleto_rate",    # boleto (cash slip) payment share
    "payment_voucher_rate",   # voucher payment share (coupon stacking proxy)
]

TEMPORAL_FEATURES: list[str] = [
    "returns_last_7",         # raw count — very recent burst
    "returns_last_30",        # raw count — 30-day window
    "returns_last_90",        # raw count — quarterly window
    "order_frequency",        # orders per month over account lifetime
    "days_since_last_order",  # recency
]

CUSTOMER_FEATURES: list[str] = [
    "account_age_days",          # tenure (short = red flag for fraud archetype)
    "avg_order_value",           # basket size
    "customer_lifetime_value",   # CLV (value net of returns)
    "product_category_entropy",  # breadth of purchase categories
    "unique_categories",         # count of distinct categories purchased
]

# Probabilistic binary flags from simulate.py — NOT deterministic thresholds
# on the label.  Each fires with archetype-specific probability, so a fraudster
# shows only SOME flags, not all.  This is the core anti-leakage design.
BINARY_FLAGS: list[str] = [
    "high_value_return",  # flag: returned a high-value item
    "coupon_heavy",       # flag: heavy coupon user
    "short_account",      # flag: account age < archetype threshold
    "burst_returner",     # flag: returns clustered in a short window
    "low_review_score",   # flag: pattern of low product review scores
    "cod_heavy",          # flag: heavy cash-on-delivery usage
]

# Derived: ratios and accelerations that expose nonlinear joint patterns.
# These are computed by _compute_derived() at feature-matrix build time.
DERIVED_FEATURES: list[str] = [
    "return_burst_rate",    # returns_last_7 / (total_orders + ε) — per-order weekly burst
    "return_acceleration",  # returns_last_30 / (returns_last_90 + ε) — recent concentration
    "clv_per_order",        # CLV / (total_orders + ε) — value efficiency signal
    "return_7d_fraction",   # returns_last_7 / (returns_last_30 + ε) — very-short-term spike
]

ALL_FEATURES: list[str] = (
    BEHAVIORAL_FEATURES
    + TEMPORAL_FEATURES
    + CUSTOMER_FEATURES
    + BINARY_FLAGS
    + DERIVED_FEATURES
)

_EPS = 1e-6   # prevents division-by-zero in derived features


# ── Derived feature computation ────────────────────────────────────────────────

def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append the four derived columns to a copy of df.

    All computations are deterministic and depend only on other columns that
    are already in the feature set — no label information touches this path.
    """
    df = df.copy()
    df["return_burst_rate"]   = df["returns_last_7"]  / (df["total_orders"]    + _EPS)
    df["return_acceleration"] = df["returns_last_30"] / (df["returns_last_90"] + _EPS)
    df["clv_per_order"]       = df["customer_lifetime_value"] / (df["total_orders"] + _EPS)
    df["return_7d_fraction"]  = df["returns_last_7"]  / (df["returns_last_30"] + _EPS)
    return df


# ── Public API ─────────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Build the raw (unscaled) feature matrix from a simulated customer DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of simulate_customers() or generate_coupon_abusers() from
        src/simulate.py.  Must contain all columns listed in ALL_FEATURES
        (after derived features are computed) plus optionally latent_type,
        binary_label, and customer_unique_id (which are excluded from X).

    Returns
    -------
    X : np.ndarray, shape (n_customers, n_features), dtype float64
        Unscaled feature array.  Safe to pass directly to XGBoost or to
        fit_scaler() / scaler.transform() before Isolation Forest / AE.
    feature_names : list[str]
        Column names in the same order as X's columns.  Length == X.shape[1].

    Notes
    -----
    - np.nan_to_num clips inf values from division-by-near-zero in derived
      features.  In practice these are rare (total_orders < 0.5 is impossible
      given the simulation floor), but the guard is cheap insurance.
    - This function is pure — it does NOT fit any transformer.  Scaling is
      the caller's responsibility (fit on train only).
    """
    df = _compute_derived(df)

    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"build_feature_matrix: missing columns in input DataFrame: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    X = df[ALL_FEATURES].values.astype(np.float64)
    # Guard: clip NaN / ±inf that can arise from near-zero denominators
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=0.0)
    return X, list(ALL_FEATURES)


# ── Scaler utilities ───────────────────────────────────────────────────────────

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """
    Fit a StandardScaler on training features and return it.

    The caller is responsible for:
      1. Calling this on X_train only (never on X_test or the full dataset).
      2. Applying scaler.transform() to X_test and any holdout set.

    StandardScaler is required by Isolation Forest and the autoencoder, both of
    which are distance/magnitude-sensitive.  XGBoost does NOT require scaling.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def save_scaler(scaler: StandardScaler, path: Path) -> None:
    """Persist a fitted scaler with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    print(f"  Scaler saved → {path}")


def load_scaler(path: Path) -> StandardScaler:
    """Load a previously saved scaler."""
    return joblib.load(path)
