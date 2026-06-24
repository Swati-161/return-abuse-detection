"""
Olist ingestion — loads the raw CSVs and aggregates to one row per customer.

Olist schema used:
  olist_orders_dataset.csv
  olist_order_items_dataset.csv
  olist_order_payments_dataset.csv
  olist_order_reviews_dataset.csv
  olist_customers_dataset.csv

Output: DataFrame with columns:
  customer_unique_id, total_orders, total_items, total_spend,
  avg_order_value, first_order_date, last_order_date,
  account_age_days, days_since_last_order, order_frequency,
  payment_type_mix_credit, payment_type_mix_boleto,
  avg_review_score
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.config import DATA_DIR, SEED


def load_olist() -> pd.DataFrame:
    """
    Load Olist CSVs and return a customer-level DataFrame.
    Raises FileNotFoundError if the data/ directory is missing the CSVs.
    """
    def _read(name: str) -> pd.DataFrame:
        path = DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Olist file: {path}\n"
                "Download from https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
            )
        return pd.read_csv(path)

    orders   = _read("olist_orders_dataset.csv")
    items    = _read("olist_order_items_dataset.csv")
    payments = _read("olist_order_payments_dataset.csv")
    reviews  = _read("olist_order_reviews_dataset.csv")
    customers = _read("olist_customers_dataset.csv")

    # ── Parse timestamps ──────────────────────────────────────────────────────
    for col in ["order_purchase_timestamp", "order_delivered_customer_date"]:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    # ── Merge customer unique ID onto orders ──────────────────────────────────
    orders = orders.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id", how="left"
    )

    # ── Order-level revenue (sum of item prices + freight) ────────────────────
    item_totals = (
        items.groupby("order_id")["price"]
             .sum()
             .reset_index()
             .rename(columns={"price": "order_value"})
    )
    orders = orders.merge(item_totals, on="order_id", how="left")

    # ── Payment mix ───────────────────────────────────────────────────────────
    pay_pivot = (
        payments.groupby(["order_id", "payment_type"])["payment_value"]
                .sum()
                .unstack(fill_value=0)
                .reset_index()
    )
    # Keep credit card and boleto; add others as 'other'
    pay_pivot.columns.name = None
    for col in ["credit_card", "boleto", "voucher", "debit_card"]:
        if col not in pay_pivot.columns:
            pay_pivot[col] = 0.0
    orders = orders.merge(pay_pivot[["order_id", "credit_card", "boleto", "voucher"]],
                          on="order_id", how="left")

    # ── Review scores ─────────────────────────────────────────────────────────
    avg_review = (
        reviews.groupby("order_id")["review_score"]
               .mean()
               .reset_index()
               .rename(columns={"review_score": "review_score"})
    )
    orders = orders.merge(avg_review, on="order_id", how="left")

    # ── Aggregate to customer level ───────────────────────────────────────────
    ref_date = orders["order_purchase_timestamp"].max()

    cust = orders.groupby("customer_unique_id").agg(
        total_orders          = ("order_id",                  "count"),
        total_spend           = ("order_value",               "sum"),
        avg_order_value       = ("order_value",               "mean"),
        first_order_date      = ("order_purchase_timestamp",  "min"),
        last_order_date       = ("order_purchase_timestamp",  "max"),
        payment_credit_total  = ("credit_card",               "sum"),
        payment_boleto_total  = ("boleto",                    "sum"),
        payment_voucher_total = ("voucher",                   "sum"),
        avg_review_score      = ("review_score",              "mean"),
    ).reset_index()

    cust["account_age_days"] = (
        ref_date - cust["first_order_date"]
    ).dt.days.clip(lower=1)

    cust["days_since_last_order"] = (
        ref_date - cust["last_order_date"]
    ).dt.days.clip(lower=0)

    cust["order_frequency"] = (
        cust["total_orders"] / (cust["account_age_days"] / 30.0)
    ).clip(lower=0)

    total_pay = (
        cust["payment_credit_total"]
        + cust["payment_boleto_total"]
        + cust["payment_voucher_total"]
    ).replace(0, np.nan)

    cust["payment_type_mix_credit"]  = cust["payment_credit_total"]  / total_pay
    cust["payment_type_mix_boleto"]  = cust["payment_boleto_total"]  / total_pay
    cust["payment_type_mix_voucher"] = cust["payment_voucher_total"] / total_pay

    cust[["payment_type_mix_credit",
          "payment_type_mix_boleto",
          "payment_type_mix_voucher"]] = (
        cust[["payment_type_mix_credit",
              "payment_type_mix_boleto",
              "payment_type_mix_voucher"]].fillna(0)
    )

    cust["avg_review_score"] = cust["avg_review_score"].fillna(3.0)
    cust["total_spend"]      = cust["total_spend"].fillna(0)
    cust["avg_order_value"]  = cust["avg_order_value"].fillna(0)

    return cust.reset_index(drop=True)


def describe_olist(df: pd.DataFrame) -> None:
    print(f"Olist customers loaded: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(df.describe().T.to_string())