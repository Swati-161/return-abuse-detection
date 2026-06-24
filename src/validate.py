"""
Phase 1 validation — confirms the simulation has genuine overlap
and the label distribution is correct.

Run this before moving to Phase 2.  If the plots look cleanly
separated, the generative model is broken (return to simulate.py).
"""

import matplotlib
matplotlib.use("Agg")   # headless — safe in any environment
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.config import PLOT_DIR, ARCHETYPES, _ARCHETYPE_ORDER


def plot_return_rate_overlap(sim: pd.DataFrame) -> None:
    """
    KDE of return_ratio per latent type.
    Should show significant overlap — if they're cleanly separated, stop.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = {"normal": "#4C72B0", "impulse": "#55A868",
              "serial": "#C44E52", "fraud": "#8172B2"}

    for t in ["normal", "impulse", "serial", "fraud"]:
        subset = sim.loc[sim["latent_type"] == t, "return_ratio"]
        subset.plot.kde(ax=ax, label=t, color=colors[t], linewidth=2)
        ax.axvspan(
            *ARCHETYPES[t]["return_rate"],
            alpha=0.06, color=colors[t]
        )

    ax.set_xlabel("return_ratio", fontsize=12)
    ax.set_ylabel("density", fontsize=12)
    ax.set_title("Return-rate distribution per latent type\n"
                 "(overlap is intentional — that's the proof the task is real)", fontsize=11)
    ax.legend()
    ax.set_xlim(-0.05, 1.05)

    path = PLOT_DIR / "return_rate_overlap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_label_distribution(sim: pd.DataFrame) -> None:
    """
    Bar chart of latent type counts and binary label split.
    Confirms class imbalance is present.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: latent types
    counts = sim["latent_type"].value_counts().reindex(
        ["normal", "impulse", "serial", "fraud"]
    )
    axes[0].bar(counts.index, counts.values,
                color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    axes[0].set_title("Latent type distribution")
    axes[0].set_ylabel("Count")
    for bar, val in zip(axes[0].patches, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{val:,} ({val/len(sim)*100:.1f}%)",
            ha="center", va="bottom", fontsize=9
        )

    # Right: binary label
    binary_counts = sim["binary_label"].value_counts().sort_index()
    axes[1].bar(["Non-fraud (0)", "Fraud (1)"], binary_counts.values,
                color=["#4C72B0", "#C44E52"])
    axes[1].set_title("Binary label distribution")
    axes[1].set_ylabel("Count")
    for bar, val in zip(axes[1].patches, binary_counts.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{val:,} ({val/len(sim)*100:.1f}%)",
            ha="center", va="bottom", fontsize=9
        )

    fig.suptitle("Dataset composition — confirming imbalance", fontsize=12)
    fig.tight_layout()

    path = PLOT_DIR / "label_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_feature_boxplots(sim: pd.DataFrame) -> None:
    """
    Box plots of key continuous features per latent type.
    Should show gradual trends with overlap, NOT clean steps.
    """
    features = [
        "return_ratio", "coupon_usage_rate", "high_value_return_rate",
        "account_age_days", "avg_days_to_return", "avg_order_value"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    order = ["normal", "impulse", "serial", "fraud"]
    for ax, feat in zip(axes, features):
        data = [sim.loc[sim["latent_type"] == t, feat].values for t in order]
        bp = ax.boxplot(data, tick_labels=order, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 2})
        colors_list = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(feat, fontsize=10)
        ax.tick_params(axis="x", labelrotation=20)

    fig.suptitle("Feature distributions per archetype\n"
                 "(gradual trends + overlap = valid simulation)", fontsize=12)
    fig.tight_layout()

    path = PLOT_DIR / "feature_boxplots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_summary(sim: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print("PHASE 1 VALIDATION SUMMARY")
    print("="*60)
    print(f"Total customers: {len(sim):,}")
    print(f"\nLatent type breakdown:")
    vc = sim["latent_type"].value_counts()
    for t, c in vc.items():
        print(f"  {t:12s}: {c:6,}  ({c/len(sim)*100:.1f}%)")
    print(f"\nBinary label: {sim['binary_label'].sum():,} fraud "
          f"({sim['binary_label'].mean()*100:.1f}%)")
    print(f"\nReturn ratio stats per type:")
    print(sim.groupby("latent_type")["return_ratio"]
            .agg(["mean", "std", "min", "max"])
            .round(3).to_string())

    # Overlap check — adjacent types should share score range
    n_rr = sim.loc[sim["latent_type"] == "normal",  "return_ratio"]
    i_rr = sim.loc[sim["latent_type"] == "impulse", "return_ratio"]
    s_rr = sim.loc[sim["latent_type"] == "serial",  "return_ratio"]
    f_rr = sim.loc[sim["latent_type"] == "fraud",   "return_ratio"]

    normal_max, impulse_min = n_rr.quantile(0.90), i_rr.quantile(0.10)
    impulse_max, serial_min = i_rr.quantile(0.90), s_rr.quantile(0.10)
    serial_max,  fraud_min  = s_rr.quantile(0.90), f_rr.quantile(0.10)

    print(f"\nOverlap check (90th pct of lower type vs 10th pct of upper):")
    print(f"  normal[90%]={normal_max:.3f}  vs  impulse[10%]={impulse_min:.3f} "
          f"  → {'OK' if normal_max > impulse_min else 'NO OVERLAP — CHECK CONFIG'}")
    print(f"  impulse[90%]={impulse_max:.3f} vs  serial[10%]={serial_min:.3f} "
          f"  → {'OK' if impulse_max > serial_min else 'NO OVERLAP — CHECK CONFIG'}")
    print(f"  serial[90%]={serial_max:.3f}  vs  fraud[10%]={fraud_min:.3f} "
          f"  → {'OK' if serial_max > fraud_min else 'NO OVERLAP — CHECK CONFIG'}")
    print("="*60 + "\n")