import numpy as np
import pandas as pd
import os

# ── PATHS ──────────────────────────────────────────────────────────────────────
RAW_PATH      = r"C:\\Users\\alexandre.batisse\\.vscode\\Projet\\Projet_Stage_Scalian\\data\\Raw\\energydata_complete.csv"
PROCESSED_DIR = r"C:\\Users\\alexandre.batisse\\.vscode\\Projet\\Projet_Stage_Scalian\\data\\Processed"
OUTPUT_CSV    = os.path.join(PROCESSED_DIR, "energydata_cleaned.csv")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── LOAD ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_PATH)
print(f"[LOAD] {df.shape[0]} rows x {df.shape[1]} columns")

# drop random noise columns — non-predictive by design
df.drop(columns=["rv1", "rv2"], errors="ignore", inplace=True)

# ── TIMESTAMPS ─────────────────────────────────────────────────────────────────
# parse date column and use it as index
df["date"] = pd.to_datetime(df["date"])
df.sort_values("date", inplace=True)
df.set_index("date", inplace=True)
df.index.name = "timestamp"

# round to nearest 10 min — fixes nanosecond drift that breaks reindex
df.index = df.index.round("10min")

# ── TEMPORAL GAPS ──────────────────────────────────────────────────────────────
# build the complete expected 10-min grid and reindex — missing steps become NaN
expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="10min")
n_gaps = len(expected_index) - len(df)
print(f"[GAPS] {n_gaps} missing steps detected and added")
df = df.reindex(expected_index)

# ── DUPLICATES ─────────────────────────────────────────────────────────────────
# keep first occurrence when the same timestamp appears twice
n_dupes = df.index.duplicated().sum()
df = df[~df.index.duplicated(keep="first")]
print(f"[DEDUP] {n_dupes} duplicates removed")

# ── MISSING VALUES ─────────────────────────────────────────────────────────────
# interpolate linearly along time (max 30 min gap)
df.interpolate(method="time", limit=3, inplace=True)
# fill leftover gaps forward then backward (max 1 h)
df.ffill(limit=6, inplace=True)
df.bfill(limit=6, inplace=True)
print(f"[MISSING] {df.isnull().sum().sum()} NaN remaining after interpolation")

# ── STUCK SENSOR VALUES ────────────────────────────────────────────────────────
# only check indoor sensors — weather columns update hourly by design
SENSOR_COLS = [c for c in df.columns if
               (c.startswith("T") and c[1:].isdigit()) or
               (c.startswith("RH_") and c[3:].isdigit())]
STUCK_THRESHOLD = 6

for col in SENSOR_COLS:
    # flag runs of identical values lasting >= STUCK_THRESHOLD steps
    groups     = (df[col] != df[col].shift()).cumsum()
    run_len    = df.groupby(groups)[col].transform("count")
    stuck_mask = (run_len >= STUCK_THRESHOLD) & (df[col] == df[col].shift())
    n = int(stuck_mask.sum())
    if n > 0:
        df.loc[stuck_mask, col] = np.nan
        df[col] = df[col].interpolate(method="time", limit=12)
        df[col] = df[col].ffill(limit=6)
        print(f"[STUCK]   {col}: {n} stuck values replaced")

# ── FINAL NaN CLEANUP ──────────────────────────────────────────────────────────
# Residual NaNs after stuck-sensor repair — gaps too long for interpolation limits
# Strategy 1 : extended time interpolation   (up to 24 h)
# Strategy 2 : same weekday × same hour median (preserves circadian patterns)
# Strategy 3 : column median                  (absolute fallback)

remaining = df.columns[df.isnull().any()].tolist()

if remaining:
    print(f"\n[FINAL_FILL] {df.isnull().sum().sum()} NaN — seasonal imputation starting...")
    for col in remaining:
        n_before = int(df[col].isnull().sum())
        if n_before == 0:
            continue

        # 1 — extended interpolation (up to 144 steps = 24 h)
        df[col] = df[col].interpolate(method="time", limit=144)
        df[col] = df[col].ffill(limit=144).bfill(limit=144)

        # 2 — same weekday + same hour median
        if df[col].isnull().any():
            seasonal = df[col].groupby(
                [df.index.dayofweek, df.index.hour]
            ).transform("median")
            df[col] = df[col].fillna(seasonal)

        # 3 — column median (last resort)
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

        n_after = int(df[col].isnull().sum())
        print(f"[FINAL_FILL]   {col}: {n_before} → {n_after} NaN")

# ── OUTLIER CAPPING ────────────────────────────────────────────────────────────
# cap all numeric columns with IQR x3 — except Appliances (target variable)
COLS_TO_CAP = [c for c in df.select_dtypes(include=np.number).columns if c != "Appliances"]
IQR_FACTOR  = 3.0

for col in COLS_TO_CAP:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR    = Q3 - Q1
    lo, hi = Q1 - IQR_FACTOR * IQR, Q3 + IQR_FACTOR * IQR
    n      = int(((df[col] < lo) | (df[col] > hi)).sum())
    if n > 0:
        df[col] = df[col].clip(lower=lo, upper=hi)
        print(f"[OUTLIER] {col}: {n} values capped [{lo:.2f}, {hi:.2f}]")

# cap Appliances separately at percentile 1–99 to preserve real peaks
lo_app = df["Appliances"].quantile(0.01)
hi_app = df["Appliances"].quantile(0.99)
df["Appliances"] = df["Appliances"].clip(lower=lo_app, upper=hi_app)
print(f"[OUTLIER] Appliances: percentile 1-99 cap [{lo_app:.1f}, {hi_app:.1f}]")

# ── PHYSICAL BOUNDS ────────────────────────────────────────────────────────────
# energy cannot be negative
for col in ["Appliances", "lights"]:
    if col in df.columns:
        df[col] = df[col].clip(lower=0)

# humidity must be between 0 and 100
for col in [c for c in df.columns if c.startswith("RH")]:
    df[col] = df[col].clip(lower=0, upper=100)

print("[BOUNDS] Energy >= 0, humidity in [0, 100]")

# ── EXPORT ─────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=True)

print()
print("=" * 50)
print("  CLEANING COMPLETED")
print("=" * 50)
print(f"  Rows           : {df.shape[0]:>10,}")
print(f"  Columns        : {df.shape[1]:>10,}")
print(f"  Remaining NaN  : {df.isnull().sum().sum():>10,}")
print(f"  Saved to       : {OUTPUT_CSV}")
print("=" * 50)
