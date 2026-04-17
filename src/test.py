import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH   = "C:\\Users\\alexandre.batisse\\.vscode\\Projet\\Projet_Stage_Scalian\\data\\Raw\\energydata_complete.csv"
OUTPUT_PATH = "C:\\Users\\alexandre.batisse\\.vscode\\Projet\\Projet_Stage_Scalian\\outputs\\Isisolation_forest_v3_scored.csv"

TRAIN_RATIO   = 0.70          # 70% des données pour l'entraînement
WINDOWS       = [6, 12, 36, 72]  # Fenêtres glissantes en périodes de 10 min
                                  # → 1h, 2h, 6h, 12h

N_ESTIMATORS  = 200
MAX_SAMPLES   = 256
RANDOM_STATE  = 42

# Seuils de sévérité calés sur les percentiles du jeu d'entraînement
# → pas de paramètre contamination biaisé
THR_LOW_Q  = 0.90   # Top 10% → anomalie "low"
THR_MID_Q  = 0.95   # Top  5% → anomalie "mid"
THR_HIGH_Q = 0.99   # Top  1% → anomalie "high"

# ── Capteurs utilisés (rv1, rv2, Visibility exclus) ──────────────────────────
ENERGY_COLS = ["Appliances", "lights"]
TEMP_IN     = ["T1","T2","T3","T4","T5","T7","T8","T9"]   # T6 = extérieur nord
HUM_IN      = ["RH_1","RH_2","RH_3","RH_4","RH_5","RH_7","RH_8","RH_9"]
TEMP_OUT    = ["T6", "T_out"]
HUM_OUT     = ["RH_6", "RH_out"]
METEO       = ["Press_mm_hg", "Windspeed", "Tdewpoint"]
ALL_SENSORS = ENERGY_COLS + TEMP_IN + HUM_IN + TEMP_OUT + HUM_OUT + METEO

# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ 1. CHARGEMENT ═══")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
df["hour"]       = df["date"].dt.hour
df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
print(f"  Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING — 7 familles de features drift-robustes
#    AUCUNE valeur brute de capteur n'est passée directement au modèle.
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ 2. FEATURE ENGINEERING ═══")
feat = pd.DataFrame(index=df.index)

# FAMILLE 1 — Z-scores glissants
# z = (valeur - moyenne_glissante_w) / std_glissante_w
# Mesure si la valeur courante est inhabituelle par rapport AUX DERNIÈRES heures,
# pas par rapport à la saison entière. Invariant au niveau absolu.
print("  Famille 1 : Z-scores glissants...")
for col in ALL_SENSORS:
    for w in WINDOWS:
        roll_mean = df[col].rolling(w, min_periods=max(2, w//2)).mean()
        roll_std  = df[col].rolling(w, min_periods=max(2, w//2)).std()
        feat[f"{col}_zscore_{w}"] = (df[col] - roll_mean) / (roll_std.replace(0, np.nan).fillna(1e-6))

# FAMILLE 2 — Taux de variation (deltas)
# delta_k = valeur(t) - valeur(t-k)
# Détecte les brusques changements indépendamment du niveau absolu.
print("  Famille 2 : Deltas...")
for col in ENERGY_COLS + TEMP_IN[:4] + HUM_IN[:4] + ["T_out", "RH_out"]:
    for lag in [1, 3, 6]:  # 10 min, 30 min, 1h
        feat[f"{col}_delta_{lag}"] = df[col].diff(lag)

# FAMILLE 3 — Déviation par rapport à la baseline heure-du-jour (sur TRAIN uniquement)
# baseline = médiane des valeurs à cette heure dans le passé (train)
# deviation = valeur - baseline_heure
# IMPORTANT : la baseline est calculée sur le train puis appliquée au test.
# Évite le data leakage et le drift (une baseline "globale" incluant le printemps
# serait biaisée si on entraîne en hiver).
# → Ce bloc est séparé après le split (voir section 4).
# Pour l'instant, on marque ces colonnes pour les recalculer après le split.
print("  Famille 3 : Déviations heure-du-jour → recalculées après split")

# FAMILLE 4 — Déviation weekend / semaine
print("  Famille 4 : Déviation weekend...")
for col in ENERGY_COLS:
    # Médiane par groupe weekend (0/1) — calculée sur tout le dataset ici car
    # c'est un comportement structurel stable (pas de drift saisonnier sur le pattern
    # weekend vs semaine).
    baseline_we = df.groupby("is_weekend")[col].transform("median")
    feat[f"{col}_we_dev"] = df[col] - baseline_we

# FAMILLE 5 — Rolling std (variabilité locale)
# rstd = écart-type sur une fenêtre glissante
# Détecte les phases anormalement calmes ou agitées.
print("  Famille 5 : Rolling std...")
for col in ENERGY_COLS + TEMP_IN[:3] + HUM_IN[:3]:
    for w in [6, 36]:
        feat[f"{col}_rstd_{w}"] = df[col].rolling(w, min_periods=2).std()

# FAMILLE 6 — Features inter-capteurs (spreads et deltas relatifs)
# Ces features sont invariantes par construction car ce sont des DIFFÉRENCES
# entre capteurs, pas des valeurs absolues.
print("  Famille 6 : Cross-capteurs...")
feat["T_indoor_spread"]         = df[TEMP_IN].max(axis=1) - df[TEMP_IN].min(axis=1)
feat["T_indoor_outdoor_delta"]  = df[TEMP_IN].mean(axis=1) - df["T_out"]
feat["T6_T_out_delta"]          = df["T6"] - df["T_out"]       # Nord ext. vs station
feat["RH_indoor_spread"]        = df[HUM_IN].max(axis=1) - df[HUM_IN].min(axis=1)
feat["RH_indoor_outdoor_delta"] = df[HUM_IN].mean(axis=1) - df["RH_out"]
feat["T1_dewpoint_margin"]      = df["T1"] - df["Tdewpoint"]   # Risque condensation
feat["energy_ratio"]            = df["lights"] / (df["Appliances"].replace(0, np.nan).fillna(1))

# FAMILLE 7 — Encodage cyclique heure et jour de la semaine
# sin/cos garantit que 23h et 0h sont proches pour le modèle.
# Invariant par nature (périodique).
print("  Famille 7 : Encodage cyclique...")
feat["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
feat["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
feat["dow_sin"]  = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
feat["dow_cos"]  = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)

print(f"  → {feat.shape[1]} features générées (avant famille 3)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SUPPRESSION DES LIGNES NaN (warmup des fenêtres glissantes)
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ 3. NETTOYAGE NaN ═══")
valid_mask  = ~feat.isna().any(axis=1)
feat_clean  = feat[valid_mask].reset_index(drop=True)
df_clean    = df[valid_mask].reset_index(drop=True)
print(f"  Lignes supprimées (warmup) : {(~valid_mask).sum()}")
print(f"  Lignes utilisables         : {len(feat_clean):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SPLIT TEMPOREL + FAMILLE 3 (baseline heure-du-jour sur train uniquement)
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ 4. SPLIT + BASELINE HEURE-DU-JOUR ═══")
split_idx  = int(len(feat_clean) * TRAIN_RATIO)
train_df   = df_clean.iloc[:split_idx].copy()

# Calcul de la baseline UNIQUEMENT sur le train
tod_baselines = {}
for col in ENERGY_COLS + TEMP_IN[:4] + ["T_out"]:
    tod_baselines[col] = {
        "median": train_df.groupby("hour")[col].median(),
        "std":    train_df.groupby("hour")[col].std().replace(0, 1e-6),
    }

# Application de la baseline sur tout le dataset (train + test)
for col in ENERGY_COLS + TEMP_IN[:4] + ["T_out"]:
    baseline_med = df_clean["hour"].map(tod_baselines[col]["median"])
    baseline_std = df_clean["hour"].map(tod_baselines[col]["std"])
    # Pour les températures, on n'ajoute que les features énergie (T saisonnières, exclues)
    if col in ENERGY_COLS:
        feat_clean[f"{col}_tod_dev"]    = df_clean[col].values - baseline_med.values
        feat_clean[f"{col}_tod_zscore"] = (df_clean[col].values - baseline_med.values) / baseline_std.values

# T_indoor_mean centré sur la moyenne train (drift saisonnier supprimé)
t_indoor_mean_train = train_df[TEMP_IN].mean(axis=1).mean()
feat_clean["T_indoor_mean_centered"] = df_clean[TEMP_IN].mean(axis=1) - t_indoor_mean_train

print(f"  Train : {split_idx:,} lignes  ({df_clean['date'].iloc[0].date()} → {df_clean['date'].iloc[split_idx-1].date()})")
print(f"  Test  : {len(feat_clean) - split_idx:,} lignes  ({df_clean['date'].iloc[split_idx].date()} → {df_clean['date'].iloc[-1].date()})")
print(f"  Features finales : {feat_clean.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. VÉRIFICATION DRIFT KS (features train vs test)
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ 5. VÉRIFICATION DRIFT KS ═══")
feat_tr = feat_clean.iloc[:split_idx]
feat_te = feat_clean.iloc[split_idx:]
ks_res  = {c: ks_2samp(feat_tr[c].values, feat_te[c].values)[0] for c in feat_clean.columns}
ks_s    = pd.Series(ks_res).sort_values(ascending=False)
severe  = (ks_s > 0.5).sum()
moderate= ((ks_s > 0.3) & (ks_s <= 0.5)).sum()
stable  = (ks_s <= 0.3).sum()
print(f"  Sévère   (KS > 0.5) : {severe}   {'⚠️' if severe else '✅ aucun'}")
print(f"  Modéré  (0.3-0.5)   : {moderate}  (tolérable)")
print(f"  Stable  (KS ≤ 0.3)  : {stable}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. NORMALISATION (StandardScaler fitté sur le train uniquement)
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ 6. NORMALISATION ═══")
X_train = feat_clean.iloc[:split_idx].copy()
X_test  = feat_clean.iloc[split_idx:].copy()

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit sur train uniquement
X_test_scaled  = scaler.transform(X_test)         # transform sur test
print(f"  Scaler fitté sur {len(X_train):,} lignes × {X_train.shape[1]} features")

# ══════════════════════════════════════════════════════════════════════════════
# 7. ENTRAÎNEMENT ISOLATION FOREST
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ 7. ENTRAÎNEMENT ISOLATION FOREST ═══")
iso = IsolationForest(
    n_estimators  = N_ESTIMATORS,
    max_samples   = MAX_SAMPLES,
    contamination = "auto",      # pas de biais via contamination
    max_features  = 1.0,
    random_state  = RANDOM_STATE,
    n_jobs        = -1,
    bootstrap     = False,
)
iso.fit(X_train_scaled)
print(f"  Modèle entraîné sur {len(X_train):,} lignes × {X_train.shape[1]} features")

# ══════════════════════════════════════════════════════════════════════════════
# 8. SCORING
# raw_score = - score_samples (plus élevé = plus anormal)
# severity_pct = rang percentile du raw_score dans la distribution train
# anomaly_category = normal / low / mid / high selon percentiles Q90/Q95/Q99
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ 8. SCORING ═══")
train_raw = -iso.score_samples(X_train_scaled)
test_raw  = -iso.score_samples(X_test_scaled)

# Seuils calés sur la distribution du TRAIN (pas sur contamination)
thr_low  = float(np.quantile(train_raw, THR_LOW_Q))
thr_mid  = float(np.quantile(train_raw, THR_MID_Q))
thr_high = float(np.quantile(train_raw, THR_HIGH_Q))
print(f"  Seuils (percentiles train) :")
print(f"    Low  (Q{int(THR_LOW_Q*100)}) : {thr_low:.4f}")
print(f"    Mid  (Q{int(THR_MID_Q*100)}) : {thr_mid:.4f}")
print(f"    High (Q{int(THR_HIGH_Q*100)}): {thr_high:.4f}")

def categorize(raw):
    cat = np.full(len(raw), "normal", dtype=object)
    cat[raw >= thr_low]  = "low"
    cat[raw >= thr_mid]  = "mid"
    cat[raw >= thr_high] = "high"
    return cat

train_sorted = np.sort(train_raw)
def severity(r):
    return np.searchsorted(train_sorted, r, side="right") / len(train_sorted)

# Construction du DataFrame de résultats
def make_results(dates, raw, split_label, df_src):
    cat = categorize(raw)
    sev = severity(raw)
    return pd.DataFrame({
        "date"             : dates,
        "split"            : split_label,
        "Appliances"       : df_src["Appliances"].values,
        "lights"           : df_src["lights"].values,
        "T1"               : df_src["T1"].values,
        "T_out"            : df_src["T_out"].values,
        "Windspeed"        : df_src["Windspeed"].values,
        "raw_score"        : raw,
        "severity_pct"     : np.round(sev * 100, 2),
        "anomaly_category" : cat,
        "anomaly_flag"     : (cat != "normal").astype(int),
    })

res_train = make_results(df_clean["date"].iloc[:split_idx].values, train_raw, "train", df_clean.iloc[:split_idx])
res_test  = make_results(df_clean["date"].iloc[split_idx:].values, test_raw,  "test",  df_clean.iloc[split_idx:])
results   = pd.concat([res_train, res_test], ignore_index=True)

print(f"\n  Distribution TEST :")
for cat, target in [("normal",90),("low",5),("mid",4),("high",1)]:
    n   = (res_test["anomaly_category"] == cat).sum()
    pct = n / len(res_test) * 100
    flag = "✅" if abs(pct - target) < 5 else "⚠️"
    print(f"    {cat:<7}: {n:>5,}  ({pct:.2f}%)  cible ≈ {target}%  {flag}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. SANITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ 9. SANITY CHECKS ═══")
ks_score, _ = ks_2samp(train_raw, test_raw)
print(f"  (a) KS(score_train, score_test) = {ks_score:.3f}  {'✅' if ks_score < 0.2 else '⚠️'}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. EXPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══ 10. EXPORT ═══")
results.to_csv(OUTPUT_PATH, index=False)
print(f"  CSV exporté → {OUTPUT_PATH}")
print(f"  Colonnes : {list(results.columns)}")
print("\n✅ DONE — Isolation Forest v3 drift-robust")