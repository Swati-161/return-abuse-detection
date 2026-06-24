"""
Central config — one place to change seeds, paths, archetype params.
"""
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data" / "olist"
OUTPUT_DIR  = ROOT / "outputs"
PLOT_DIR    = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42

# ── Simulation size ────────────────────────────────────────────────────────────
# How many synthetic customers to generate ON TOP of the real Olist scaffold.
# Olist has ~99k orders / ~96k unique customers; we use those as a base and
# augment with synthetic records so the total dataset is large enough to
# train on minority classes.
N_SYNTHETIC_EXTRA = 3_000   # extra fully-synthetic rows beyond the Olist base

# ── Archetype definitions ──────────────────────────────────────────────────────
# return_rate: (lo, hi) for the Normal distribution mean.
# The std is (hi - lo) / 3 so ±1σ covers the range — ensures overlap.
ARCHETYPES = {
    "normal":  {"weight": 0.70, "return_rate": (0.05, 0.15)},
    "impulse": {"weight": 0.18, "return_rate": (0.10, 0.35)},
    "serial":  {"weight": 0.09, "return_rate": (0.30, 0.70)},
    "fraud":   {"weight": 0.03, "return_rate": (0.50, 0.95)},
}

_ARCHETYPE_ORDER = [
    "normal",
    "impulse",
    "serial",
    "fraud",
]

# Coupon-abuser archetype — HELD OUT, never in training.
# Deliberately different signature from fraud (moderate return rate, long tenure).
COUPON_ABUSER = {
    "return_rate":        (0.08, 0.22),   # NOT high — different from fraud
    "coupon_usage_rate":  (0.70, 0.95),   # the defining signal
    "account_age_days":   (400, 900),     # long-tenured — NOT short like fraud
    "avg_order_value":    (30, 90),       # modest
}

# ── Label noise ───────────────────────────────────────────────────────────────
# Fraction of customers whose latent type is randomly re-assigned after
# all features are generated. This builds in irreducible error on purpose.
LABEL_NOISE_RATE = 0.04   # 4 %

# ── Binary target ─────────────────────────────────────────────────────────────
# Version 1: fraudster vs everyone else.
FRAUD_CLASS = "fraud"