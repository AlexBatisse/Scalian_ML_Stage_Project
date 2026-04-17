import os
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

RAW_PATH      = r"C:\Users\alexandre.batisse\.vscode\Projet\Projet_Stage_Scalian\data\Raw\energydata_complete.csv"
PROCESSED_DIR = r"C:\Users\alexandre.batisse\.vscode\Projet\Projet_Stage_Scalian\data\Processed"
OUTPUT_CSV    = os.path.join(PROCESSED_DIR, "energydata_anomaly_ready.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

FREQ            = "10min"    # native dataset frequency
STEPS_PER_HOUR  = 6          # 6 × 10 min = 1 h
STEPS_PER_DAY   = 144        # 144 × 10 min = 24 h

# Rolling window sizes (in steps)
ROLL_1H   = STEPS_PER_HOUR         # 6
ROLL_6H   = STEPS_PER_HOUR * 6    # 36
ROLL_24H  = STEPS_PER_DAY         # 144

# Max gap to fill via interpolation (conservative — 30 min = 3 steps)
# Longer gaps are intentionally left as NaN or flagged — they may be anomalies
MAX_FILL_STEPS = 3

# Stuck sensor: how many identical consecutive steps before flagging
STUCK_THRESHOLD = 18  # 3 hour of identical readings
STUCK_MIN_RANGE  = 0.05

# Nighttime hours (inclusive)
NIGHT_START = 22
NIGHT_END   = 6
TRAIN_RATIO = 0.70 

# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

def safe_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling Z-score with guard against zero std (stuck sensor / constant zone)."""
    roll  = series.rolling(window=window, min_periods=window // 2)
    mu    = roll.mean()
    sigma = roll.std()
    # Where std == 0, z-score is ill-defined; set to 0 (no deviation)
    z     = (series - mu) / sigma.replace(0, np.nan)
    return z.fillna(0)


def safe_deviation(series: pd.Series, window: int) -> pd.Series:
    """(value - rolling_median) / rolling_std  — robust local deviation."""
    roll   = series.rolling(window=window, min_periods=window // 2)
    median = roll.median()
    sigma  = roll.std()
    dev    = (series - median) / sigma.replace(0, np.nan)
    return dev.fillna(0)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & BASIC CLEANUP
# ══════════════════════════════════════════════════════════════════════════════
section("1 · LOAD")

df = pd.read_csv(RAW_PATH)
print(f"  Loaded  : {df.shape[0]:,} rows × {df.shape[1]} columns")

# Drop noise columns — non-informative by construction
df.drop(columns=["rv1", "rv2"], errors="ignore", inplace=True)
print(f"  Dropped : rv1, rv2")

# Parse & sort timestamps
df["date"] = pd.to_datetime(df["date"])
df.sort_values("date", inplace=True)
df.set_index("date", inplace=True)
df.index.name = "timestamp"
df.index = df.index.round(FREQ)   # fix nanosecond drift

# ── Duplicates ──────────────────────────────────────────────────────────────
n_dupes = df.index.duplicated().sum()
df = df[~df.index.duplicated(keep="first")]
print(f"  Dupes   : {n_dupes} removed")

# ── Temporal gaps — reindex to complete 10-min grid ─────────────────────────
expected = pd.date_range(start=df.index.min(), end=df.index.max(), freq=FREQ)
n_gaps   = len(expected) - len(df)
df       = df.reindex(expected)
print(f"  Gaps    : {n_gaps} missing steps materialised as NaN")

# ══════════════════════════════════════════════════════════════════════════════
# 2. CONSERVATIVE GAP FILLING
# ══════════════════════════════════════════════════════════════════════════════
section("2 · CONSERVATIVE GAP FILLING  (max 30 min)")

# Only fill very short gaps (≤ MAX_FILL_STEPS = 3 steps = 30 min)
# Longer gaps are intentionally preserved — they may represent real anomalies
# (e.g. sensor outage, power cut) and should NOT be imputed away.

before = int(df.isnull().sum().sum())
df.interpolate(method="time", limit=MAX_FILL_STEPS, inplace=True)
df.ffill(limit=MAX_FILL_STEPS, inplace=True)
df.bfill(limit=MAX_FILL_STEPS, inplace=True)
after  = int(df.isnull().sum().sum())
print(f"  NaN before : {before:,}")
print(f"  NaN after  : {after:,}  ({before - after:,} filled)")
print(f"  NaN kept   : {after:,}  (long gaps — intentional)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. STUCK SENSOR FLAGS  (no replacement — a stuck sensor IS an anomaly)
# ══════════════════════════════════════════════════════════════════════════════
section("3 · STUCK SENSOR FLAGS")

SENSOR_COLS = [c for c in df.columns if
               (c.startswith("T") and c[1:].isdigit()) or
               (c.startswith("RH_") and c[3:].isdigit())]

total_stuck = 0
for col in SENSOR_COLS:
    groups   = (df[col] != df[col].shift()).cumsum()
    run_len  = df.groupby(groups)[col].transform("count")
    # Also check the rolling range on a 6h window: if the sensor varies normally
    # around the stuck point (sub-resolution noise absent), it's physical stability,
    # not a fault. Flag only when value is locked AND surrounding window is flat.
    local_range = (df[col].rolling(window=ROLL_6H, min_periods=ROLL_6H // 2).max() -
                   df[col].rolling(window=ROLL_6H, min_periods=ROLL_6H // 2).min())
    flag_col = f"flag_stuck_{col}"
    df[flag_col] = (
        (run_len >= STUCK_THRESHOLD) &
        (df[col] == df[col].shift()) &
        (local_range < STUCK_MIN_RANGE)
    ).astype(int)

    n = int(df[flag_col].sum())
    total_stuck += n
    if n > 0:
        print(f"  {col:<10} : {n:>5} stuck steps flagged → '{flag_col}'")

print(f"  Total stuck steps flagged : {total_stuck:,}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PHYSICAL BOUNDS  (hard physical limits only — NOT outlier capping)
# ══════════════════════════════════════════════════════════════════════════════
section("4 · PHYSICAL BOUNDS")

# Energy consumption cannot be negative (physics)
for col in ["Appliances", "lights"]:
    if col in df.columns:
        neg = int((df[col] < 0).sum())
        df[col] = df[col].clip(lower=0)
        print(f"  {col:<15}: {neg} negative values set to 0")

# Humidity must be in [0, 100] — a reading of 110% is a sensor fault
for col in [c for c in df.columns if c.startswith("RH")]:
    out = int(((df[col] < 0) | (df[col] > 100)).sum())
    df[col] = df[col].clip(lower=0, upper=100)
    if out > 0:
        print(f"  {col:<15}: {out} values clipped to [0, 100]")

# ══════════════════════════════════════════════════════════════════════════════
# 5. TEMPORAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════
section("5 · TEMPORAL FEATURES")

df["hour_of_day"]  = df.index.hour
df["day_of_week"]  = df.index.dayofweek          # 0=Mon … 6=Sun
df["month"]        = df.index.month
df["day_of_year"]  = df.index.dayofyear
df["week_of_year"] = df.index.isocalendar().week.astype(int)

# Boolean flags
df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
df["is_nighttime"] = ((df["hour_of_day"] >= NIGHT_START) |
                      (df["hour_of_day"] <  NIGHT_END)).astype(int)

# Cyclical encoding — helps ML models understand periodicity of hour & day
df["hour_sin"]   = np.sin(2 * np.pi * df["hour_of_day"] / 24)
df["hour_cos"]   = np.cos(2 * np.pi * df["hour_of_day"] / 24)
df["dow_sin"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)

print(f"  Added : hour_of_day, day_of_week, month, day_of_year, week_of_year")
print(f"          is_weekend, is_nighttime")
print(f"          cyclical sin/cos for hour, day_of_week, month")

# ══════════════════════════════════════════════════════════════════════════════
# 6. LAG FEATURES
# ══════════════════════════════════════════════════════════════════════════════
section("6 · LAG FEATURES")

# Apply to energy and key sensor columns
LAG_COLS = ["Appliances", "lights", "T1", "T_out", "RH_1", "RH_out"]
LAGS     = {
    "lag1"  : 1,               # 10 min ago
    "lag6"  : STEPS_PER_HOUR,  # 1 hour ago
    "lag144": STEPS_PER_DAY,   # 24 hours ago
}

for col in LAG_COLS:
    if col not in df.columns:
        continue
    for lag_name, lag_steps in LAGS.items():
        df[f"{col}_{lag_name}"] = df[col].shift(lag_steps)

lag_cols_added = [f"{c}_{l}" for c in LAG_COLS for l in LAGS if c in df.columns]
print(f"  Added {len(lag_cols_added)} lag columns")
print(f"  Cols : {', '.join(LAG_COLS)}")
print(f"  Lags : t-1 (10min), t-6 (1h), t-144 (24h)")

# ══════════════════════════════════════════════════════════════════════════════
# 7. RATE OF CHANGE (DELTA)
# ══════════════════════════════════════════════════════════════════════════════
section("7 · RATE OF CHANGE")

DELTA_COLS = ["Appliances", "lights", "T1", "T2", "T_out", "RH_1", "RH_out"]

for col in DELTA_COLS:
    if col not in df.columns:
        continue
    # Absolute delta (consecutive 10-min step)
    df[f"{col}_delta"]    = df[col].diff()
    # Relative delta over 1h (more stable signal)
    df[f"{col}_delta_1h"] = df[col].diff(STEPS_PER_HOUR)

delta_cols_added = [f"{c}_delta" for c in DELTA_COLS if c in df.columns] + \
                   [f"{c}_delta_1h" for c in DELTA_COLS if c in df.columns]
print(f"  Added {len(delta_cols_added)} delta columns  (Δt=10min + Δt=1h)")

# ══════════════════════════════════════════════════════════════════════════════
# 8. ROLLING STATISTICS (mean + std)
# ══════════════════════════════════════════════════════════════════════════════
section("8 · ROLLING STATISTICS  (1h / 6h / 24h)")

# Apply to all original numeric sensor / energy columns
NUMERIC_ORIG = (
    ["Appliances", "lights"] +
    [c for c in df.columns if c.startswith("T")  and c[1:].isdigit()] +
    [c for c in df.columns if c.startswith("RH_") and c[3:].isdigit()] +
    ["T_out", "RH_out", "Press_mm_hg", "Windspeed", "Visibility", "Tdewpoint"]
)
NUMERIC_ORIG = [c for c in NUMERIC_ORIG if c in df.columns]

ROLL_WINDOWS = {"1h": ROLL_1H, "6h": ROLL_6H, "24h": ROLL_24H}

roll_added = 0
for col in NUMERIC_ORIG:
    if col not in df.columns:
        continue
    for wname, wsize in ROLL_WINDOWS.items():
        roll = df[col].rolling(window=wsize, min_periods=wsize // 2)
        df[f"{col}_rmean_{wname}"] = roll.mean()
        df[f"{col}_rstd_{wname}"]  = roll.std()
        roll_added += 2

print(f"  Added {roll_added} rolling columns across {len(NUMERIC_ORIG)} base columns")
print(f"  Windows : 1h ({ROLL_1H} steps) / 6h ({ROLL_6H}) / 24h ({ROLL_24H})")

# ══════════════════════════════════════════════════════════════════════════════
# 9. ROLLING Z-SCORE
# ══════════════════════════════════════════════════════════════════════════════
section("9 · ROLLING Z-SCORE  (24h window)")

# Z-score relative to the last 24h — main anomaly signal for threshold-based models
# Use 24h window to capture daily cycle baseline
ZSCORE_COLS = (["Appliances", "lights"] +
               [c for c in df.columns if c.startswith("T") and not c.startswith("T_") and c[1:].isdigit()] +
               [c for c in df.columns if c.startswith("RH_") and c[3:].isdigit()] +
               ["T_out", "RH_out"])

for col in ZSCORE_COLS:
    if col not in df.columns:
        continue
    df[f"{col}_zscore_24h"] = safe_zscore(df[col], ROLL_24H)
    df[f"{col}_zscore_6h"]  = safe_zscore(df[col], ROLL_6H)

zscore_added = sum(1 for c in ZSCORE_COLS if c in df.columns) * 2
print(f"  Added {zscore_added} Z-score columns (24h + 6h baseline)")

# ══════════════════════════════════════════════════════════════════════════════
# 10. DEVIATION FROM LOCAL PATTERN
# ══════════════════════════════════════════════════════════════════════════════
section("10 · DEVIATION FROM LOCAL PATTERN  (value - rolling_median) / rolling_std")

# Robust to outliers — uses median not mean
DEV_COLS = ["Appliances", "T1", "T_out", "RH_1"]

for col in DEV_COLS:
    if col not in df.columns:
        continue
    df[f"{col}_dev_1h"]  = safe_deviation(df[col], ROLL_1H)
    df[f"{col}_dev_24h"] = safe_deviation(df[col], ROLL_24H)

dev_added = sum(1 for c in DEV_COLS if c in df.columns) * 2
print(f"  Added {dev_added} deviation columns for: {', '.join(DEV_COLS)}")

# ══════════════════════════════════════════════════════════════════════════════
# 11. CROSS-SENSOR FEATURES
# ══════════════════════════════════════════════════════════════════════════════
section("11 · CROSS-SENSOR FEATURES")

# Temperature gradient indoor vs outdoor
# T1 = kitchen, T2 = living room — representative indoor zones
if "T1" in df.columns and "T_out" in df.columns:
    df["T_indoor_outdoor_delta"] = df["T1"] - df["T_out"]
    print(f"  T_indoor_outdoor_delta = T1 - T_out")

if "T2" in df.columns and "T_out" in df.columns:
    df["T2_outdoor_delta"] = df["T2"] - df["T_out"]
    print(f"  T2_outdoor_delta = T2 - T_out")

# Indoor temperature spread (max - min across rooms) — alerts if one room drifts
T_indoor_cols = [c for c in df.columns if c.startswith("T") and c[1:].isdigit()]
if len(T_indoor_cols) >= 2:
    df["T_indoor_spread"] = df[T_indoor_cols].max(axis=1) - df[T_indoor_cols].min(axis=1)
    print(f"  T_indoor_spread = max(T1..T9) - min(T1..T9)  [room drift indicator]")

# Humidity gradient indoor vs outdoor
if "RH_1" in df.columns and "RH_out" in df.columns:
    df["RH_indoor_outdoor_delta"] = df["RH_1"] - df["RH_out"]
    print(f"  RH_indoor_outdoor_delta = RH_1 - RH_out")

# Indoor humidity spread
RH_indoor_cols = [c for c in df.columns if c.startswith("RH_") and c[3:].isdigit()]
if len(RH_indoor_cols) >= 2:
    df["RH_indoor_spread"] = df[RH_indoor_cols].max(axis=1) - df[RH_indoor_cols].min(axis=1)
    print(f"  RH_indoor_spread  [humidity uniformity indicator]")

# Dew point spread (condensation risk)
if "Tdewpoint" in df.columns and "T1" in df.columns:
    df["T1_dewpoint_margin"] = df["T1"] - df["Tdewpoint"]
    print(f"  T1_dewpoint_margin = T1 - Tdewpoint  [condensation risk]")

# ══════════════════════════════════════════════════════════════════════════════
# 12. ENERGY SIGNATURE  (Appliances vs expected for this time-of-day)
# ══════════════════════════════════════════════════════════════════════════════
section("12 · ENERGY SIGNATURE")

# Expected consumption = median for this exact (hour, day_of_week) slot
# over the entire dataset history — gives contextual baseline
# Cutoff index: use only the first TRAIN_RATIO of rows to learn the baseline
_cutoff  = int(len(df) * TRAIN_RATIO)
_train   = df.iloc[:_cutoff]

# Learn baselines on TRAIN ONLY, then map back onto the full timeline
_tod_med_map = _train.groupby(["hour_of_day","day_of_week"])["Appliances"].median()
_tod_std_map = _train.groupby(["hour_of_day","day_of_week"])["Appliances"].std()
_key = list(zip(df["hour_of_day"], df["day_of_week"]))
tod_median = pd.Series([_tod_med_map.get(k, np.nan) for k in _key], index=df.index)
tod_std    = pd.Series([_tod_std_map.get(k, np.nan) for k in _key], index=df.index)

df["Appliances_tod_median"]    = tod_median
df["Appliances_tod_deviation"] = df["Appliances"] - tod_median
df["Appliances_tod_zscore"]    = (
    (df["Appliances"] - tod_median) / tod_std.replace(0, np.nan)
).fillna(0)

# Weekend vs weekday signature — same logic, fit on train
_we_map = _train.groupby(["hour_of_day","is_weekend"])["Appliances"].median()
_key_we = list(zip(df["hour_of_day"], df["is_weekend"]))
tod_we_median = pd.Series([_we_map.get(k, np.nan) for k in _key_we], index=df.index)
df["Appliances_we_deviation"] = df["Appliances"] - tod_we_median

print(f"  Baselines learnt on {_cutoff:,} train rows ({TRAIN_RATIO:.0%})")

print("  Appliances_tod_median    : expected consumption for (hour, day_of_week)")
print("  Appliances_tod_deviation : actual - expected")
print("  Appliances_tod_zscore    : normalised deviation from expected")
print("  Appliances_we_deviation  : actual - expected  (weekend/weekday split)")

# ══════════════════════════════════════════════════════════════════════════════
# 13. SEASONAL DECOMPOSITION RESIDUALS  (STL on Appliances)
# ══════════════════════════════════════════════════════════════════════════════
section("13 · SEASONAL DECOMPOSITION RESIDUALS  (STL)")

# STL decomposes the signal into trend + seasonal + residuals
# Residuals = what's left after removing expected patterns → pure anomaly signal
# Period = 144 steps = 24h (daily seasonality)

print("  Running STL decomposition on Appliances (period=144)...")

# STL requires no NaN — fill temporarily for decomposition only
# Fit STL on TRAIN portion only to avoid leaking test-period seasonality
app_series = df["Appliances"].ffill().bfill()
app_train  = app_series.iloc[:_cutoff]

try:
    stl    = STL(app_train, period=STEPS_PER_DAY, robust=True)
    result = stl.fit()

    # Build a per-(hour, day_of_week) seasonal template from the fit
    seasonal_train = pd.Series(result.seasonal.values, index=app_train.index)
    seasonal_template = seasonal_train.groupby(
        [seasonal_train.index.hour, seasonal_train.index.dayofweek]
    ).median()
    _key_stl = list(zip(df.index.hour, df.index.dayofweek))
    df["Appliances_stl_seasonal"] = [seasonal_template.get(k, 0.0) for k in _key_stl]

    # Trend: keep train trend, extend with rolling median on the test portion
    trend_full = pd.Series(np.nan, index=df.index)
    trend_full.iloc[:_cutoff] = result.trend.values
    trend_full = trend_full.fillna(
        df["Appliances"].rolling(window=ROLL_24H, min_periods=ROLL_24H // 2).median()
    )
    df["Appliances_stl_trend"] = trend_full

    # Residual = actual - trend - seasonal (computable at inference time)
    df["Appliances_stl_residual"] = (
        df["Appliances"] - df["Appliances_stl_trend"] - df["Appliances_stl_seasonal"]
    )

    # Z-score normalised by TRAIN residual std (no test leakage)
    res_std = result.resid.std()
    df["Appliances_stl_resid_zscore"] = (
        df["Appliances_stl_residual"] / res_std if res_std > 0
        else pd.Series(0, index=df.index)
    )

except Exception as e:
    print(f"  ⚠️  STL failed ({e}) — fallback to rolling-median residuals")
    roll_24 = df["Appliances"].rolling(window=ROLL_24H, min_periods=ROLL_24H // 2).median()
    df["Appliances_stl_residual"]      = df["Appliances"] - roll_24
    df["Appliances_stl_resid_zscore"]  = safe_zscore(df["Appliances_stl_residual"], ROLL_24H)
    df["Appliances_stl_trend"]         = roll_24
    df["Appliances_stl_seasonal"]      = np.nan

# ══════════════════════════════════════════════════════════════════════════════
# 14. NaN AUDIT & FINAL EXPORT
# ══════════════════════════════════════════════════════════════════════════════
section("14 · NaN AUDIT")

nan_report = df.isnull().sum()
nan_report = nan_report[nan_report > 0].sort_values(ascending=False)

if not nan_report.empty:
    print(f"  {'Column':<40} {'NaN count':>10}  {'% of rows':>10}")
    print(f"  {'─'*62}")
    for col, cnt in nan_report.items():
        pct = cnt / len(df) * 100
        origin = ("← lag/delta warmup" if any(x in col for x in ["lag", "delta", "rmean", "rstd"])
                  else "← structural gap"   if pct < 5
                  else "← long gap / expected")
        print(f"  {col:<40} {cnt:>10,}  {pct:>9.1f}%  {origin}")
else:
    print("  ✅ No NaN remaining")

print(f"\n  Total NaN  : {int(df.isnull().sum().sum()):,}")
print(f"  Total rows : {len(df):,}")
print(f"  Total cols : {len(df.columns):,}")

# ── Export ──────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=True)

# Column catalogue
print(f"\n{'═'*60}")
print("  COLUMN CATALOGUE")
print(f"{'═'*60}")
groups = {
    "Raw sensors (indoor temp)"    : [c for c in df.columns if c.startswith("T") and c[1:].isdigit()],
    "Raw sensors (indoor humidity)": [c for c in df.columns if c.startswith("RH_") and c[3:].isdigit()],
    "Raw sensors (weather)"        : ["T_out","Press_mm_hg","RH_out","Windspeed","Visibility","Tdewpoint"],
    "Raw energy"                   : ["Appliances","lights"],
    "Stuck sensor flags"           : [c for c in df.columns if c.startswith("flag_stuck")],
    "Temporal features"            : [c for c in df.columns if c in
                                       ["hour_of_day","day_of_week","month","day_of_year",
                                        "week_of_year","is_weekend","is_nighttime",
                                        "hour_sin","hour_cos","dow_sin","dow_cos",
                                        "month_sin","month_cos"]],
    "Lag features"                 : [c for c in df.columns if "_lag" in c],
    "Rate of change"               : [c for c in df.columns if "_delta" in c],
    "Rolling mean"                 : [c for c in df.columns if "_rmean_" in c],
    "Rolling std"                  : [c for c in df.columns if "_rstd_" in c],
    "Rolling Z-score"              : [c for c in df.columns if "_zscore_" in c and "tod" not in c and "stl" not in c],
    "Local deviation"              : [c for c in df.columns if "_dev_" in c],
    "Cross-sensor"                 : [c for c in df.columns if any(x in c for x in
                                       ["_delta","_spread","_margin"]) and "_1h" not in c and "stl" not in c
                                       and c not in [cc for cc in df.columns if "_delta" in cc and any(
                                       s in cc for s in ["Appliances","lights","T1","T2","T_out","RH_1","RH_out"])]],
    "Energy signature"             : [c for c in df.columns if "tod" in c or "we_deviation" in c],
    "STL decomposition"            : [c for c in df.columns if "stl" in c],
}
for group, cols in groups.items():
    cols = [c for c in cols if c in df.columns]
    if cols:
        print(f"\n  ▸ {group}  ({len(cols)} columns)")
        for c in cols:
            print(f"    {c}")

print(f"\n{'═'*60}")
print("  EXPORT COMPLETED")
print(f"{'═'*60}")
print(f"  Rows    : {len(df):>10,}")
print(f"  Columns : {len(df.columns):>10,}")
print(f"  NaN     : {int(df.isnull().sum().sum()):>10,}")
print(f"  Saved → {OUTPUT_CSV}")
print(f"{'═'*60}\n")