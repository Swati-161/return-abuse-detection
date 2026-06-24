"""
Generative simulation layer.

Takes the Olist customer scaffold and assigns each customer a latent archetype,
then samples their behavioral features from type-conditional distributions
with deliberate overlap and noise.

Key design principles (from brief §2):
  - The latent type is the HIDDEN cause, never a threshold on any feature.
  - Features are SAMPLED with variance, not set to fixed archetype values.
  - Adjacent archetypes have overlapping distributions — that's intentional.
  - ~4% label noise is added after generation to build in irreducible error.
  - The coupon-abuser archetype is generated separately and NEVER mixed into
    the training population.
"""

import numpy as np
import pandas as pd
from src.config import (
    ARCHETYPES, COUPON_ABUSER, SEED,
    LABEL_NOISE_RATE, FRAUD_CLASS, N_SYNTHETIC_EXTRA
)


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sample_bounded(rng: np.random.Generator,
                    lo: float, hi: float,
                    n: int = 1,
                    clip_lo: float = 0.0,
                    clip_hi: float = 1.0) -> np.ndarray:
    """
    Sample n values from Normal(mean=(lo+hi)/2, std=(hi-lo)/3).
    std = range/3 so ±1σ covers the intended range, ensuring adjacent
    archetypes overlap where their ranges touch.
    """
    mean = (lo + hi) / 2
    std  = (hi - lo) / 3
    vals = rng.normal(mean, std, size=n)
    return np.clip(vals, clip_lo, clip_hi)


def _bernoulli(rng: np.random.Generator, p: float, n: int) -> np.ndarray:
    return (rng.random(n) < p).astype(float)


# ──────────────────────────────────────────────────────────────────────────────
#  Per-archetype probability tables for binary behavioral signals
# ──────────────────────────────────────────────────────────────────────────────
#
# These drive the probabilistic red-flag indicators.
# Each flag fires independently with archetype-specific probability.
# No single flag deterministically defines a class.
#
# Columns: normal / impulse / serial / fraud
_P = {
    #  signal                     normal  impulse  serial  fraud
    "high_value_return":        [ 0.05,   0.15,    0.40,   0.70 ],
    "coupon_heavy":             [ 0.10,   0.20,    0.50,   0.80 ],
    "short_account":            [ 0.10,   0.20,    0.30,   0.60 ],
    "burst_returner":           [ 0.03,   0.10,    0.35,   0.65 ],
    "low_review_score":         [ 0.10,   0.20,    0.45,   0.60 ],
    "cod_heavy":                [ 0.15,   0.25,    0.40,   0.55 ],
}

_ARCHETYPE_ORDER = ["normal", "impulse", "serial", "fraud"]


def _archetype_index(t: str) -> int:
    return _ARCHETYPE_ORDER.index(t)


# ──────────────────────────────────────────────────────────────────────────────
#  Core: generate behavioral features for a batch of customers of type t
# ──────────────────────────────────────────────────────────────────────────────

def _generate_batch(rng: np.random.Generator,
                    archetype: str,
                    n: int,
                    base_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Generate n customers of the given archetype.

    If base_df is provided (a slice of Olist rows), the generated features
    augment the real transaction scaffold.  Otherwise, all columns are synthetic.
    """
    ai = _archetype_index(archetype)
    cfg = ARCHETYPES[archetype]

    # ── Core return rate (the most important signal; sampled, not thresholded) ─
    return_ratio = _sample_bounded(rng, *cfg["return_rate"], n=n)

    # ── Total orders — positively skewed, type-conditional ────────────────────
    order_mu = {"normal": 3.0, "impulse": 4.5, "serial": 6.0, "fraud": 4.0}[archetype]
    total_orders = np.clip(
        rng.poisson(lam=order_mu, size=n), 1, 40
    ).astype(float)

    total_returns = np.clip(
        np.round(return_ratio * total_orders), 0, total_orders
    )

    # ── High-value return rate ─────────────────────────────────────────────────
    high_value_return_rate = _sample_bounded(
        rng,
        lo=0.0,
        hi={"normal": 0.10, "impulse": 0.25, "serial": 0.55, "fraud": 0.85}[archetype],
        n=n, clip_lo=0.0, clip_hi=1.0
    )

    # ── Coupon usage rate ──────────────────────────────────────────────────────
    coupon_usage_rate = _sample_bounded(
        rng,
        lo={"normal": 0.0, "impulse": 0.05, "serial": 0.20, "fraud": 0.40}[archetype],
        hi={"normal": 0.15, "impulse": 0.30, "serial": 0.65, "fraud": 0.90}[archetype],
        n=n, clip_lo=0.0, clip_hi=1.0
    )

    # ── Avg days to return ─────────────────────────────────────────────────────
    avg_days_to_return = np.clip(
        rng.normal(
            loc={"normal": 20, "impulse": 14, "serial": 8, "fraud": 4}[archetype],
            scale=6,
            size=n
        ), 1, 60
    )

    # ── Account age (days) ─────────────────────────────────────────────────────
    # short account = red flag for fraud
    if base_df is not None:
        account_age_days = base_df["account_age_days"].values.copy()
    else:
        age_lo = {"normal": 200, "impulse": 100, "serial": 80, "fraud": 10}[archetype]
        age_hi = {"normal": 900, "impulse": 600, "serial": 400, "fraud": 200}[archetype]
        account_age_days = _sample_bounded(
            rng, age_lo, age_hi, n=n,
            clip_lo=1.0, clip_hi=1200.0
        )

    # ── Average order value ────────────────────────────────────────────────────
    if base_df is not None:
        avg_order_value = base_df["avg_order_value"].values.copy()
    else:
        aov_lo = {"normal": 30, "impulse": 50, "serial": 80, "fraud": 120}[archetype]
        aov_hi = {"normal": 150, "impulse": 250, "serial": 350, "fraud": 600}[archetype]
        avg_order_value = _sample_bounded(
            rng, aov_lo, aov_hi, n=n,
            clip_lo=5.0, clip_hi=2000.0
        )

    # ── Temporal features ──────────────────────────────────────────────────────
    order_frequency = np.clip(
        rng.exponential(
            scale={"normal": 0.5, "impulse": 1.2, "serial": 1.8, "fraud": 1.0}[archetype],
            size=n
        ), 0.01, 10.0
    )

    returns_last_30 = np.clip(
        np.round(return_ratio * total_orders *
                 rng.uniform(0.2, 0.5, size=n)),
        0, total_returns
    )
    returns_last_7 = np.clip(
        np.round(returns_last_30 * rng.uniform(0.1, 0.4, size=n)),
        0, returns_last_30
    )
    returns_last_90 = np.clip(
        total_returns * rng.uniform(0.7, 1.0, size=n),
        returns_last_30, total_returns
    )

    days_since_last = np.clip(
        rng.exponential(
            scale={"normal": 90, "impulse": 45, "serial": 30, "fraud": 60}[archetype],
            size=n
        ), 0, 730
    )

    # ── Customer lifetime value ────────────────────────────────────────────────
    customer_lifetime_value = avg_order_value * total_orders * (
        1 - 0.3 * return_ratio
    )

    # ── Product category diversity (entropy-like) ──────────────────────────────
    # Fraudsters tend to narrow-target high-value categories
    product_category_entropy = _sample_bounded(
        rng,
        lo={"normal": 0.5, "impulse": 0.4, "serial": 0.3, "fraud": 0.1}[archetype],
        hi={"normal": 1.0, "impulse": 0.9, "serial": 0.7, "fraud": 0.5}[archetype],
        n=n, clip_lo=0.0, clip_hi=1.0
    )
    unique_categories = np.clip(
        np.round(product_category_entropy * 8 + 1), 1, 10
    ).astype(float)

    # ── Payment mix (COD heavy = slight fraud signal in some markets) ──────────
    if base_df is not None:
        payment_boleto = base_df["payment_type_mix_boleto"].values.copy()
        payment_voucher = base_df["payment_type_mix_voucher"].values.copy()
    else:
        payment_boleto = _bernoulli(
            rng, {"normal": 0.15, "impulse": 0.20, "serial": 0.30, "fraud": 0.40}[archetype], n
        ) * rng.uniform(0.3, 0.8, n)
        payment_voucher = _bernoulli(
            rng, {"normal": 0.08, "impulse": 0.15, "serial": 0.40, "fraud": 0.70}[archetype], n
        ) * rng.uniform(0.2, 0.9, n)

    # ── Binary red-flag signals ────────────────────────────────────────────────
    p_row = [_P[sig][ai] for sig in _P]
    binary_flags = {
        sig: _bernoulli(rng, _P[sig][ai], n)
        for sig in _P
    }

    df = pd.DataFrame({
        "total_orders":             total_orders,
        "total_returns":            total_returns,
        "return_ratio":             return_ratio,
        "high_value_return_rate":   high_value_return_rate,
        "coupon_usage_rate":        coupon_usage_rate,
        "avg_days_to_return":       avg_days_to_return,
        "account_age_days":         account_age_days,
        "avg_order_value":          avg_order_value,
        "order_frequency":          order_frequency,
        "returns_last_7":           returns_last_7,
        "returns_last_30":          returns_last_30,
        "returns_last_90":          returns_last_90,
        "days_since_last_order":    days_since_last,
        "customer_lifetime_value":  customer_lifetime_value,
        "product_category_entropy": product_category_entropy,
        "unique_categories":        unique_categories,
        "payment_boleto_rate":      payment_boleto,
        "payment_voucher_rate":     payment_voucher,
        **binary_flags,
        "latent_type":              archetype,
    })

    return df


# ──────────────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────────────

def simulate_customers(olist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns each Olist customer a latent archetype and overlays simulated
    behavioral features on top of the real transaction scaffold.

    Returns the full customer dataset with latent_type and binary_label columns.
    """
    rng = np.random.default_rng(SEED)
    n = len(olist_df)

    # ── Assign latent types ───────────────────────────────────────────────────
    weights = [ARCHETYPES[t]["weight"] for t in _ARCHETYPE_ORDER]
    types = rng.choice(_ARCHETYPE_ORDER, size=n, p=weights)

    # ── Generate features per type (batched for efficiency) ───────────────────
    frames = []
    for archetype in _ARCHETYPE_ORDER:
        mask = types == archetype
        count = mask.sum()
        if count == 0:
            continue
        base_slice = olist_df[mask].reset_index(drop=True)
        batch = _generate_batch(rng, archetype, count, base_df=base_slice)
        batch.index = olist_df.index[mask]
        frames.append(batch)

    sim = pd.concat(frames).sort_index().reset_index(drop=True)

    # ── Carry over Olist ID and any remaining real columns ────────────────────
    sim["customer_unique_id"] = olist_df["customer_unique_id"].values

    # ── Add ~4% label noise (flip latent type for a small random subset) ──────
    noise_mask = rng.random(n) < LABEL_NOISE_RATE
    noise_indices = np.where(noise_mask)[0]
    for idx in noise_indices:
        current = sim.at[idx, "latent_type"]
        others = [t for t in _ARCHETYPE_ORDER if t != current]
        sim.at[idx, "latent_type"] = rng.choice(others)

    # ── Binary target (V1: fraudster vs everyone else) ────────────────────────
    sim["binary_label"] = (sim["latent_type"] == FRAUD_CLASS).astype(int)

    return sim


def generate_coupon_abusers(n: int = 500) -> pd.DataFrame:
    """
    Generate the HELD-OUT coupon-abuser population.

    NEVER mix this into training data.  Used only in the held-out
    archetype evaluation experiment (Phase 6).
    """
    rng = np.random.default_rng(SEED + 999)
    cfg = COUPON_ABUSER

    return_ratio = _sample_bounded(rng, *cfg["return_rate"], n=n)
    coupon_usage_rate = _sample_bounded(rng, *cfg["coupon_usage_rate"], n=n)
    account_age_days = _sample_bounded(
        rng, *cfg["account_age_days"], n=n, clip_lo=30, clip_hi=1500
    )
    avg_order_value = _sample_bounded(
        rng, *cfg["avg_order_value"], n=n, clip_lo=5, clip_hi=500
    )
    total_orders = np.clip(rng.poisson(4.0, n), 1, 30).astype(float)
    total_returns = np.clip(np.round(return_ratio * total_orders), 0, total_orders)

    df = pd.DataFrame({
        "total_orders":             total_orders,
        "total_returns":            total_returns,
        "return_ratio":             return_ratio,
        "high_value_return_rate":   _sample_bounded(rng, 0.0, 0.15, n=n),
        "coupon_usage_rate":        coupon_usage_rate,
        "avg_days_to_return":       np.clip(rng.normal(18, 6, n), 1, 60),
        "account_age_days":         account_age_days,
        "avg_order_value":          avg_order_value,
        "order_frequency":          np.clip(rng.exponential(0.6, n), 0.01, 5),
        "returns_last_7":           np.zeros(n),
        "returns_last_30":          np.clip(np.round(total_returns * 0.3), 0, total_returns),
        "returns_last_90":          total_returns,
        "days_since_last_order":    np.clip(rng.exponential(60, n), 0, 500),
        "customer_lifetime_value":  avg_order_value * total_orders * 0.9,
        "product_category_entropy": _sample_bounded(rng, 0.5, 0.9, n=n),
        "unique_categories":        np.clip(rng.poisson(4, n), 1, 10).astype(float),
        "payment_boleto_rate":      np.clip(rng.uniform(0.0, 0.2, n), 0, 1),
        "payment_voucher_rate":     np.clip(rng.uniform(0.5, 0.9, n), 0, 1),
        # binary flags
        "high_value_return":        (rng.random(n) < 0.05).astype(float),
        "coupon_heavy":             (rng.random(n) < 0.90).astype(float),
        "short_account":            (rng.random(n) < 0.05).astype(float),
        "burst_returner":           (rng.random(n) < 0.05).astype(float),
        "low_review_score":         (rng.random(n) < 0.10).astype(float),
        "cod_heavy":                (rng.random(n) < 0.10).astype(float),
        "latent_type":              "coupon_abuser",
        "binary_label":             0,   # NOT labeled as fraud — that's the point
    })

    return df