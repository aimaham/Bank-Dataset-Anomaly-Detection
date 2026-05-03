"""
Askari Bank — Transaction Anomaly Detection
Uses Isolation Forest with customer-contextual feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# ── 1. Load data ──────────────────────────────────────────────────────────────
txn = pd.read_csv("transactions.csv", parse_dates=["datetime"])
profiles = pd.read_csv("customer_profiles.csv")

txn = txn.merge(
    profiles[["customer_id", "declared_monthly_income_pkr", "account_type", "occupation"]],
    on="customer_id",
    how="left",
)

# ── 2. Time-based features ────────────────────────────────────────────────────
txn["hour"]        = txn["datetime"].dt.hour
txn["day_of_week"] = txn["datetime"].dt.dayofweek  # 0=Monday … 6=Sunday
txn["is_weekend"]  = (txn["day_of_week"] >= 5).astype(int)

# Off-hours: before 8 AM or after 10 PM — unusual for branch/ATM activity
txn["is_off_hours"] = ((txn["hour"] < 8) | (txn["hour"] >= 22)).astype(int)

# ── 3. Customer-level aggregate features (historical context) ─────────────────
# These let the model understand "large for THIS customer" vs. large in absolute terms.
cust_stats = (
    txn.groupby("customer_id")["amount_pkr"]
    .agg(cust_mean="mean", cust_std="std", cust_median="median")
    .reset_index()
)
cust_stats["cust_std"] = cust_stats["cust_std"].fillna(0)

txn = txn.merge(cust_stats, on="customer_id", how="left")

# Amount relative to customer's own history
txn["amount_vs_mean"]   = txn["amount_pkr"] / (txn["cust_mean"] + 1)
txn["amount_vs_median"] = txn["amount_pkr"] / (txn["cust_median"] + 1)

# Z-score within customer — how many std deviations above the customer's norm
txn["amount_zscore"] = (txn["amount_pkr"] - txn["cust_mean"]) / (txn["cust_std"] + 1)

# Amount relative to declared monthly income (catches implausible single transactions)
txn["income_ratio"] = txn["amount_pkr"] / (txn["declared_monthly_income_pkr"].replace(0, 1))

# ── 4. Transaction velocity (rolling count per customer per day) ──────────────
txn = txn.sort_values(["customer_id", "datetime"])
txn["txn_date"] = txn["datetime"].dt.date

daily_count = (
    txn.groupby(["customer_id", "txn_date"])
    .cumcount() + 1
)
txn["daily_txn_count"] = daily_count.values

# ── 5. City mismatch flag (transaction city ≠ home city from profile) ─────────
profiles_city = profiles[["customer_id", "city"]].rename(columns={"city": "home_city"})
txn = txn.merge(profiles_city, on="customer_id", how="left")
txn["city_mismatch"] = (txn["city"] != txn["home_city"]).astype(int)

# ── 6. Encode categoricals ────────────────────────────────────────────────────
# Label-encoding is sufficient here; Isolation Forest only uses numeric splits
# and doesn't assume ordinality in the same way a linear model would.
le = LabelEncoder()
txn["txn_type_enc"] = le.fit_transform(txn["txn_type"])
txn["channel_enc"]  = le.fit_transform(txn["channel"])
txn["city_enc"]     = le.fit_transform(txn["city"])

# ── 7. Assemble feature matrix ────────────────────────────────────────────────
FEATURES = [
    # Raw amount + relative-to-customer signals
    "amount_pkr",
    "amount_vs_mean",
    "amount_vs_median",
    "amount_zscore",
    "income_ratio",
    # Time signals
    "hour",
    "day_of_week",
    "is_weekend",
    "is_off_hours",
    # Velocity
    "daily_txn_count",
    # Geography
    "city_mismatch",
    # Encoded categoricals
    "txn_type_enc",
    "channel_enc",
    "city_enc",
]

X = txn[FEATURES].fillna(0).astype(float)

# ── 8. Isolation Forest ───────────────────────────────────────────────────────
# contamination=0.02 (2 %):
#   Real-world banking fraud rates sit at roughly 0.1–1 % of transactions,
#   but this dataset is synthetic and designed to include a handful of planted
#   anomalies. Setting 2 % gives a small safety margin above the typical 1 %
#   floor while staying well below the 5–10 % defaults that generate too many
#   false positives for an operations team to review. At ~740 rows, 2 % flags
#   ≈15 transactions — a reviewable workload for a compliance officer.
model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42,
)
model.fit(X)

txn["anomaly_flag"]  = model.predict(X)          # -1 = anomaly, 1 = normal
txn["anomaly_score"] = model.score_samples(X)    # lower = more anomalous

# ── 9. Top 10 most suspicious transactions ────────────────────────────────────
top10 = (
    txn.sort_values("anomaly_score")
    .head(10)[
        [
            "txn_id", "customer_id", "datetime", "amount_pkr",
            "txn_type", "channel", "city", "home_city",
            "anomaly_score", "amount_zscore", "income_ratio",
            "hour", "is_off_hours", "city_mismatch", "daily_txn_count",
            "note",
        ]
    ]
    .reset_index(drop=True)
)

# ── 10. Print results ─────────────────────────────────────────────────────────
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.3f}".format)

print("=" * 80)
print("TOP 10 MOST ANOMALOUS TRANSACTIONS")
print("=" * 80)
print(top10.to_string(index=True))
print()

print("-" * 80)
print("PER-TRANSACTION NARRATIVE")
print("-" * 80)
for _, row in top10.iterrows():
    flags = []
    if row["amount_zscore"] > 2:
        flags.append(f"amount {row['amount_zscore']:.1f} std-devs above customer mean")
    if row["income_ratio"] > 1:
        flags.append(f"amount is {row['income_ratio']:.1f}x declared monthly income")
    if row["is_off_hours"]:
        flags.append(f"off-hours ({int(row['hour']):02d}:xx)")
    if row["city_mismatch"]:
        flags.append(f"city mismatch (home={row['home_city']}, txn={row['city']})")
    if row["daily_txn_count"] > 5:
        flags.append(f"high daily velocity ({int(row['daily_txn_count'])} txns that day)")

    print(
        f"  {row['txn_id']}  {row['customer_id']}  "
        f"PKR {row['amount_pkr']:>12,.0f}  {row['txn_type']:<20s}  "
        f"score={row['anomaly_score']:.4f}"
    )
    if flags:
        print(f"    >> {'; '.join(flags)}")
    if row["note"] and str(row["note"]) not in ("nan", ""):
        print(f"    >> note: {row['note']}")
    print()
