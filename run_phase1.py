"""
Phase 1 runner.

Usage:
    python run_phase1.py

Outputs:
    outputs/customers_simulated.parquet   — full dataset, ready for Phase 2
    outputs/coupon_abusers.parquet        — held-out evaluation set
    outputs/plots/*.png                   — validation plots
"""

import pandas as pd
from src.ingest import load_olist, describe_olist
from src.simulate import simulate_customers, generate_coupon_abusers
from src.validate import (
    plot_return_rate_overlap,
    plot_label_distribution,
    plot_feature_boxplots,
    print_summary,
)
from src.config import OUTPUT_DIR, SEED
import numpy as np


def main():
    print("── Phase 1: Olist ingestion ──────────────────────────────")
    olist = load_olist()
    describe_olist(olist)

    print("\n── Generative simulation ─────────────────────────────────")
    sim = simulate_customers(olist)

    print("\n── Validation ────────────────────────────────────────────")
    print_summary(sim)
    plot_return_rate_overlap(sim)
    plot_label_distribution(sim)
    plot_feature_boxplots(sim)

    print("\n── Generating held-out coupon abusers ────────────────────")
    coupon_abusers = generate_coupon_abusers(n=500)
    print(f"Coupon abusers: {len(coupon_abusers):,} rows")

    print("\n── Saving outputs ────────────────────────────────────────")
    out_path = OUTPUT_DIR / "customers_simulated.parquet"
    sim.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    coupon_path = OUTPUT_DIR / "coupon_abusers.parquet"
    coupon_abusers.to_parquet(coupon_path, index=False)
    print(f"Saved: {coupon_path}")

    print("\n✓ Phase 1 complete. Check outputs/plots/ before proceeding to Phase 2.")


if __name__ == "__main__":
    main()