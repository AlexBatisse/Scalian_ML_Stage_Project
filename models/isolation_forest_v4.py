import os
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
import shap

warnings.filterwarnings("ignore")


# =============================================================================
# 0. CONFIGURATION
# =============================================================================
DATA_PATH  = "C:\\Users\\alexandre.batisse\\.vscode\\Projet\\Projet_Stage_Scalian\\data\\Raw\\energydata_complete.csv"
OUTPUT_DIR = "C:\\Users\\alexandre.batisse\\.vscode\\Projet\\Projet_Stage_Scalian\\outputs\output_Isolation_forest"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_RATIO  = 0.70
WINDOWS      = [6, 12, 36, 72] # périodes de 10 min → 1h, 2h, 6h, 12h
N_ESTIMATORS = 200
MAX_SAMPLES  = 256
RANDOM_STATE = 42

THR_LOW_Q  = 0.90
THR_MID_Q  = 0.95
THR_HIGH_Q = 0.99

ENERGY_COLS = ["Appliances", "lights"]
TEMP_IN     = ["T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9"]
HUM_IN      = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_7", "RH_8", "RH_9"]
TEMP_OUT    = ["T6", "T_out"]
HUM_OUT     = ["RH_6", "RH_out"]
METEO       = ["Press_mm_hg", "Windspeed", "Tdewpoint"]
ALL_SENSORS = ENERGY_COLS + TEMP_IN + HUM_IN + TEMP_OUT + HUM_OUT + METEO

# Mapping capteur physique → label lisible
# Adapte ces labels a ta connaissance du batiment etude
SENSOR_LABELS = {
    "Appliances" : "Consommation appareils (Wh)",
    "lights"     : "Eclairage (Wh)",
    "T1"         : "Temperature cuisine (C)",
    "T2"         : "Temperature salon/salle de vie (C)",
    "T3"         : "Temperature buanderie (C)",
    "T4"         : "Temperature bureau (C)",
    "T5"         : "Temperature salle de bain (C)",
    "T6"         : "Temperature ext. Nord (C)",
    "T7"         : "Temperature chambre parentale (C)",
    "T8"         : "Temperature salon ado (C)",
    "T9"         : "Temperature chambre (C)",
    "T_out"      : "Temperature exterieure station meteo (C)",
    "RH_1"       : "Humidite cuisine (%)",
    "RH_2"       : "Humidite salon/salle de vie (%)",
    "RH_3"       : "Humidite buanderie (%)",
    "RH_4"       : "Humidite bureau (%)",
    "RH_5"       : "Humidite salle de bain (%)",
    "RH_6"       : "Humidite ext. Nord (%)",
    "RH_7"       : "Humidite chambre parentale (%)",
    "RH_8"       : "Humidite salon ado (%)",
    "RH_9"       : "Humidite chambre (%)",
    "RH_out"     : "Humidite exterieure (%)",
    "Press_mm_hg": "Pression atmospherique (mmHg)",
    "Windspeed"  : "Vitesse du vent (m/s)",
    "Tdewpoint"  : "Point de rosee (C)",
}


# =============================================================================
# 1. CHARGEMENT
# =============================================================================
print("\n=== 1. CHARGEMENT ===")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
df["hour"]       = df["date"].dt.hour
df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
print(f"  Dataset : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")
print(f"  Periode : {df['date'].min()} -> {df['date'].max()}")


# =============================================================================
# 2. FEATURE ENGINEERING — 7 familles drift-robustes
# =============================================================================
print("\n=== 2. FEATURE ENGINEERING ===")
feat = pd.DataFrame(index=df.index)

# FAMILLE 1 — Z-scores glissants
# z = (x - mean_fenetre) / std_fenetre
# Invariant au niveau absolu : si tout est "normalement" plus chaud en mai,
# le z-score reste autour de 0. Seules les déviations LOCALES ressortent.

print("  [1/7] Z-scores glissants...")
for col in ALL_SENSORS:
    for w in WINDOWS:
        rm = df[col].rolling(w, min_periods=max(2, w // 2)).mean()
        rs = df[col].rolling(w, min_periods=max(2, w // 2)).std()
        feat[f"{col}_zscore_{w}"] = (df[col] - rm) / rs.replace(0, np.nan).fillna(1e-6)


# FAMILLE 2 — Deltas (taux de variation)
# delta_k = x(t) - x(t-k)  → brusques changements indépendants du niveau

print("  [2/7] Deltas...")
for col in ENERGY_COLS + TEMP_IN[:4] + HUM_IN[:4] + ["T_out", "RH_out"]:
    for lag in [1, 3, 6]:
        feat[f"{col}_delta_{lag}"] = df[col].diff(lag)

# FAMILLE 3 — Déviation heure-du-jour (calculée sur le TRAIN uniquement, cf section 4)

print("  [3/7] Deviations heure-du-jour -> reportees apres split (section 4)")


# FAMILLE 4 — Déviation weekend / semaine

print("  [4/7] Deviation weekend...")
for col in ENERGY_COLS:
    baseline_we = df.groupby("is_weekend")[col].transform("median")
    feat[f"{col}_we_dev"] = df[col] - baseline_we

# FAMILLE 5 — Rolling std (variabilité locale)
# rstd élevé = phase agitée ; rstd très bas = phase anormalement calme

print("  [5/7] Rolling std...")
for col in ENERGY_COLS + TEMP_IN[:3] + HUM_IN[:3]:
    for w in [6, 36]:
        feat[f"{col}_rstd_{w}"] = df[col].rolling(w, min_periods=2).std()

# FAMILLE 6 — Spreads inter-capteurs (différences entre capteurs = toujours relatif)

print("  [6/7] Cross-capteurs...")
feat["T_indoor_spread"]         = df[TEMP_IN].max(axis=1) - df[TEMP_IN].min(axis=1)
feat["T_indoor_outdoor_delta"]  = df[TEMP_IN].mean(axis=1) - df["T_out"]
feat["T6_T_out_delta"]          = df["T6"] - df["T_out"]
feat["RH_indoor_spread"]        = df[HUM_IN].max(axis=1) - df[HUM_IN].min(axis=1)
feat["RH_indoor_outdoor_delta"] = df[HUM_IN].mean(axis=1) - df["RH_out"]
feat["T1_dewpoint_margin"]      = df["T1"] - df["Tdewpoint"]
feat["energy_ratio"]            = df["lights"] / df["Appliances"].replace(0, np.nan).fillna(1)

# FAMILLE 7 — Encodage cyclique heure / jour
# sin/cos : 23h et 0h sont "proches" pour le modèle

print("  [7/7] Encodage cyclique...")
feat["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
feat["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
feat["dow_sin"]  = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
feat["dow_cos"]  = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)

print(f"  -> {feat.shape[1]} features generees (famille 3 reportee)")


# =============================================================================
# 3. NETTOYAGE NaN
# =============================================================================
print("\n=== 3. NETTOYAGE NaN ===")
valid_mask = ~feat.isna().any(axis=1)
feat_clean = feat[valid_mask].reset_index(drop=True)
df_clean   = df[valid_mask].reset_index(drop=True)
print(f"  Lignes supprimees (warmup) : {(~valid_mask).sum()}")
print(f"  Lignes utilisables         : {len(feat_clean):,}")


# =============================================================================
# 4. SPLIT TEMPOREL + FAMILLE 3
# =============================================================================
print("\n=== 4. SPLIT + BASELINE HEURE-DU-JOUR ===")
split_idx = int(len(feat_clean) * TRAIN_RATIO)
train_df  = df_clean.iloc[:split_idx].copy()

tod_baselines = {}
for col in ENERGY_COLS:
    tod_baselines[col] = {
        "median": train_df.groupby("hour")[col].median(),
        "std"   : train_df.groupby("hour")[col].std().replace(0, 1e-6),
    }

for col in ENERGY_COLS:
    bmed = df_clean["hour"].map(tod_baselines[col]["median"])
    bstd = df_clean["hour"].map(tod_baselines[col]["std"])
    feat_clean[f"{col}_tod_dev"]    = df_clean[col].values - bmed.values
    feat_clean[f"{col}_tod_zscore"] = (df_clean[col].values - bmed.values) / bstd.values

t_mean_train = train_df[TEMP_IN].mean(axis=1).mean()
feat_clean["T_indoor_mean_centered"] = df_clean[TEMP_IN].mean(axis=1) - t_mean_train

FEATURE_NAMES = list(feat_clean.columns)
print(f"  Train : {split_idx:,} lignes  ({df_clean['date'].iloc[0].date()} -> {df_clean['date'].iloc[split_idx-1].date()})")
print(f"  Test  : {len(feat_clean)-split_idx:,} lignes  ({df_clean['date'].iloc[split_idx].date()} -> {df_clean['date'].iloc[-1].date()})")
print(f"  Features finales : {len(FEATURE_NAMES)}")


# =============================================================================
# 5. VERIFICATION DRIFT KS
# =============================================================================
print("\n=== 5. VERIFICATION DRIFT KS ===")
feat_tr = feat_clean.iloc[:split_idx]
feat_te = feat_clean.iloc[split_idx:]
ks_res  = {c: ks_2samp(feat_tr[c].values, feat_te[c].values)[0] for c in FEATURE_NAMES}
ks_s    = pd.Series(ks_res).sort_values(ascending=False)
print(f"  Severe   (KS > 0.5) : {(ks_s > 0.5).sum()}")
print(f"  Modere   (0.3-0.5)  : {((ks_s > 0.3) & (ks_s <= 0.5)).sum()}")
print(f"  Stable   (KS <= 0.3): {(ks_s <= 0.3).sum()}")
print(f"  KS max              : {ks_s.max():.3f}  {'[OK]' if ks_s.max() < 0.30 else '[ATTENTION]'}")


# =============================================================================
# 6. NORMALISATION
# =============================================================================
print("\n=== 6. NORMALISATION ===")
X_train = feat_clean.iloc[:split_idx].values
X_test  = feat_clean.iloc[split_idx:].values

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"  Scaler fitte sur {len(X_train):,} lignes")


# =============================================================================
# 7. ENTRAINEMENT
# =============================================================================
print("\n=== 7. ENTRAINEMENT ISOLATION FOREST ===")
iso = IsolationForest(
    n_estimators  = N_ESTIMATORS,
    max_samples   = MAX_SAMPLES,
    contamination = "auto",
    max_features  = 1.0,
    random_state  = RANDOM_STATE,
    n_jobs        = -1,
    bootstrap     = False,
)
iso.fit(X_train_scaled)
print(f"  Modele entraine : {N_ESTIMATORS} arbres x {MAX_SAMPLES} echantillons")


# =============================================================================
# 8. SCORING
# =============================================================================
print("\n=== 8. SCORING ===")
train_raw = -iso.score_samples(X_train_scaled)
test_raw  = -iso.score_samples(X_test_scaled)

thr_low  = float(np.quantile(train_raw, THR_LOW_Q))
thr_mid  = float(np.quantile(train_raw, THR_MID_Q))
thr_high = float(np.quantile(train_raw, THR_HIGH_Q))
print(f"  Seuils : Low={thr_low:.4f} (Q{int(THR_LOW_Q*100)}) | Mid={thr_mid:.4f} (Q{int(THR_MID_Q*100)}) | High={thr_high:.4f} (Q{int(THR_HIGH_Q*100)})")

def categorize(raw):
    cat = np.full(len(raw), "normal", dtype=object)
    cat[raw >= thr_low]  = "low"
    cat[raw >= thr_mid]  = "mid"
    cat[raw >= thr_high] = "high"
    return cat

train_sorted = np.sort(train_raw)
def severity(r):
    return np.searchsorted(train_sorted, r, side="right") / len(train_sorted)

test_cat = categorize(test_raw)
test_sev = severity(test_raw)

print(f"\n  Distribution TEST :")
for cat, target in [("normal", 90), ("low", 5), ("mid", 4), ("high", 1)]:
    n   = (test_cat == cat).sum()
    pct = 100 * n / len(test_cat)
    flag = "[OK]" if abs(pct - target) < 5 else "[ATTENTION]"
    print(f"    {cat:<7}: {n:>5,}  ({pct:.2f}%)  cible ~{target}%  {flag}")


# =============================================================================
# 9. TEST SUR 100% DU DATASET
# =============================================================================
print("\n=== 9. TEST SUR 100% DU DATASET ===")
X_full_scaled = scaler.transform(feat_clean.values)
full_raw      = -iso.score_samples(X_full_scaled)
full_cat      = categorize(full_raw)
full_sev      = severity(full_raw)

full_test_cat = full_cat[split_idx:]
accord        = np.mean(test_cat == full_test_cat)

print(f"\n  Distribution GLOBALE (100%) :")
for cat in ["normal", "low", "mid", "high"]:
    n   = (full_cat == cat).sum()
    pct = 100 * n / len(full_cat)
    print(f"    {cat:<7}: {n:>6,}  ({pct:.1f}%)")


# =============================================================================
# 10. EXPORT CSV SCORED (7030 + FULL)
# =============================================================================
print("\n=== 10. EXPORT CSV SCORED ===")

def make_result_df(dates, raw, sev, cat, df_src, split_label):
    return pd.DataFrame({
        "date"             : dates,
        "split"            : split_label,
        "Appliances"       : df_src["Appliances"].values,
        "lights"           : df_src["lights"].values,
        "T1"               : df_src["T1"].values,
        "T2"               : df_src["T2"].values,
        "T_out"            : df_src["T_out"].values,
        "RH_1"             : df_src["RH_1"].values,
        "RH_out"           : df_src["RH_out"].values,
        "Windspeed"        : df_src["Windspeed"].values,
        "raw_score"        : raw,
        "severity_pct"     : np.round(sev * 100, 2),
        "anomaly_category" : cat,
        "anomaly_flag"     : (cat != "normal").astype(int),
    })

res_train = make_result_df(
    df_clean["date"].iloc[:split_idx].values,
    train_raw, severity(train_raw), categorize(train_raw),
    df_clean.iloc[:split_idx], "train"
)
res_test = make_result_df(
    df_clean["date"].iloc[split_idx:].values,
    test_raw, test_sev, test_cat,
    df_clean.iloc[split_idx:], "test"
)
res_7030 = pd.concat([res_train, res_test], ignore_index=True)

res_full = make_result_df(
    df_clean["date"].values, full_raw, full_sev, full_cat, df_clean, "full"
)


path_full = os.path.join(OUTPUT_DIR, "isolation_forest_v4_scored_full.csv")

res_full.to_csv(path_full, index=False)
print(f"  Exporte : {path_full}")


# =============================================================================
# 11. SHAP — AGREGATION PAR CAPTEUR PHYSIQUE
#
# Objectif : relier les valeurs SHAP des features engineered aux capteurs
# bruts du CSV d'origine, afin que le resultat final soit lisible sans
# connaitre les features internes du modele.
#
# Methode :
#   1. Pour chaque feature engineered (ex: T2_rstd_6), on identifie le
#      capteur physique d'origine (ici : T2).
#   2. On additionne les valeurs SHAP absolues de toutes les features
#      appartenant au meme capteur → "score de responsabilite" du capteur.
#   3. On joint les valeurs brutes du CSV original (T2=23.4 C) ainsi que
#      les percentiles contextuels (vs distribution normale train).
#   4. On exporte 1 ligne par anomalie mid/high avec :
#      - Contexte horodatage, heure, jour, saison
#      - Valeurs brutes des capteurs-cles
#      - Top 5 capteurs les plus responsables avec leur score SHAP agrege
#      - Interpretation automatique par capteur (regle metier simple)
# =============================================================================
print("\n=== 11. SHAP — ANALYSE PAR CAPTEUR PHYSIQUE ===")

# --- 11.1 Correspondance feature engineered -> capteur physique ---
# Chaque feature est rattachee a un capteur source.
# Les features cross-capteurs (ex: T_indoor_spread) sont rattachees
# a la categorie "multi" et decomposees sur leurs capteurs constituants.

def feature_to_sensor(feat_name):
    """
    Retourne le(s) capteur(s) physique(s) associe(s) a une feature engineered.
    Retourne une liste de (capteur, poids_relatif).
    """
    f = str(feat_name)

    # Features mono-capteur : format "CAPTEUR_suffixe"
    for sensor in ALL_SENSORS:
        if f.startswith(sensor + "_"):
            return [(sensor, 1.0)]

    # Features cross-capteurs
    cross_map = {
        "T_indoor_spread"        : [(s, 1/len(TEMP_IN)) for s in TEMP_IN],
        "T_indoor_outdoor_delta" : [(s, 1/len(TEMP_IN)) for s in TEMP_IN] + [("T_out", 0.1)],
        "T6_T_out_delta"         : [("T6", 0.5), ("T_out", 0.5)],
        "RH_indoor_spread"       : [(s, 1/len(HUM_IN)) for s in HUM_IN],
        "RH_indoor_outdoor_delta": [(s, 1/len(HUM_IN)) for s in HUM_IN] + [("RH_out", 0.1)],
        "T1_dewpoint_margin"     : [("T1", 0.5), ("Tdewpoint", 0.5)],
        "energy_ratio"           : [("lights", 0.5), ("Appliances", 0.5)],
        "T_indoor_mean_centered" : [(s, 1/len(TEMP_IN)) for s in TEMP_IN],
    }
    if f in cross_map:
        return cross_map[f]

    # Features cycliques → pas de capteur physique, ignorer
    if f in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        return [("_temporal_", 1.0)]

    return [("_other_", 1.0)]


# Precalcul : pour chaque feature, sa liste (capteur, poids)
feat_sensor_map = {f: feature_to_sensor(f) for f in FEATURE_NAMES}

# --- 11.2 Calcul SHAP ---
np.random.seed(RANDOM_STATE)
bg_idx    = np.where(categorize(train_raw) == "normal")[0]
bg_sample = X_train_scaled[np.random.choice(bg_idx, 150, replace=False)]

# Anomalies mid + high sur le jeu de test complet (pas de sous-echantillonnage)
explain_mask     = np.isin(full_cat, ["high", "mid"])
X_explain        = X_full_scaled[explain_mask]
df_explain       = df_clean[explain_mask].reset_index(drop=True)
raw_explain      = full_raw[explain_mask]
cat_explain      = full_cat[explain_mask]
sev_explain      = full_sev[explain_mask]

print(f"  Anomalies mid+high a expliquer : {len(X_explain)}")
print("  Calcul SHAP KernelExplainer...")



def model_fn(X):
    return -iso.score_samples(X)

explainer = shap.KernelExplainer(model_fn, bg_sample)

# nsamples=200 : bon compromis vitesse/precision
# Pour plus de precision : nsamples=500 (plus lent)
shap_vals = explainer.shap_values(X_explain, nsamples=200, silent=True)
print(f"  SHAP calcule : shape={shap_vals.shape}")

# --- 11.3 Calcul des percentiles contextuels par capteur (sur les normaux du train) ---
# Pour chaque capteur, on calcule le percentile de la valeur brute de l'anomalie
# par rapport a la distribution des observations normales du train.
# Cela permet de savoir si la valeur brute est haute, basse ou dans la norme.
print("  Calcul des percentiles contextuels...")

train_normal_mask = categorize(train_raw) == "normal"
df_train_normal   = df_clean.iloc[:split_idx][train_normal_mask]

# Percentiles par heure du jour (contexte horaire)
# On compare la valeur de l'anomalie a la distribution normale de la meme heure
sensor_pct_by_hour = {}
for sensor in ALL_SENSORS:
    sensor_pct_by_hour[sensor] = {}
    for h in range(24):
        vals = df_train_normal[df_train_normal["hour"] == h][sensor].values
        sensor_pct_by_hour[sensor][h] = np.sort(vals) if len(vals) > 0 else np.array([0])

def get_contextual_percentile(sensor, value, hour):
    """Percentile de 'value' dans la distribution normale de ce capteur a cette heure."""
    ref = sensor_pct_by_hour.get(sensor, {}).get(hour, np.array([0]))
    if len(ref) == 0:
        return 50.0
    return float(np.searchsorted(ref, value, side="right") / len(ref) * 100)

# --- 11.4 Agregation SHAP par capteur + construction du CSV final ---
print("  Agregation SHAP par capteur physique et construction du rapport...")

rows = []
for i in range(len(X_explain)):
    sv = shap_vals[i]   # vecteur SHAP pour cette anomalie, longueur = nb features

    # Agregation : score de responsabilite par capteur
    sensor_scores = {}
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        shap_abs = abs(sv[feat_idx])
        if shap_abs == 0:
            continue
        shap_signed = sv[feat_idx]
        for (sensor, weight) in feat_sensor_map[feat_name]:
            if sensor in ("_temporal_", "_other_"):
                continue
            if sensor not in sensor_scores:
                sensor_scores[sensor] = {"abs": 0.0, "signed": 0.0, "n_feat": 0}
            sensor_scores[sensor]["abs"]    += shap_abs * weight
            sensor_scores[sensor]["signed"] += shap_signed * weight
            sensor_scores[sensor]["n_feat"] += 1

    # Tri par responsabilite absolue decroissante
    sorted_sensors = sorted(sensor_scores.items(), key=lambda x: x[1]["abs"], reverse=True)

    # Valeurs brutes de ce point
    row_raw  = df_explain.iloc[i]
    hour_val = int(row_raw["hour"])

    # Construction de la ligne de sortie
    row_out = {
        "date"              : row_raw["date"],
        "anomaly_category"  : cat_explain[i],
        "severity_pct"      : round(sev_explain[i] * 100, 2),
        "raw_score"         : round(float(raw_explain[i]), 6),
        "hour"              : hour_val,
        "day_of_week"       : row_raw["date"].day_name(),
        "is_weekend"        : int(row_raw["is_weekend"]),
        # Valeurs brutes des capteurs principaux
        "Appliances_Wh"     : round(float(row_raw["Appliances"]), 1),
        "lights_Wh"         : round(float(row_raw["lights"]), 1),
        "T1_C"              : round(float(row_raw["T1"]), 2),
        "T2_C"              : round(float(row_raw["T2"]), 2),
        "T3_C"              : round(float(row_raw["T3"]), 2),
        "T4_C"              : round(float(row_raw["T4"]), 2),
        "T5_C"              : round(float(row_raw["T5"]), 2),
        "T6_C"              : round(float(row_raw["T6"]), 2),
        "T7_C"              : round(float(row_raw["T7"]), 2),
        "T8_C"              : round(float(row_raw["T8"]), 2),
        "T9_C"              : round(float(row_raw["T9"]), 2),
        "T_out_C"           : round(float(row_raw["T_out"]), 2),
        "RH_1_pct"          : round(float(row_raw["RH_1"]), 1),
        "RH_2_pct"          : round(float(row_raw["RH_2"]), 1),
        "RH_3_pct"          : round(float(row_raw["RH_3"]), 1),
        "RH_5_pct"          : round(float(row_raw["RH_5"]), 1),
        "RH_out_pct"        : round(float(row_raw["RH_out"]), 1),
        "Windspeed_ms"      : round(float(row_raw["Windspeed"]), 2),
        "Press_mmHg"        : round(float(row_raw["Press_mm_hg"]), 1),
        "Tdewpoint_C"       : round(float(row_raw["Tdewpoint"]), 2),
    }

    # Top 5 capteurs responsables
    for rank, (sensor, scores) in enumerate(sorted_sensors[:5], start=1):
        raw_val = float(row_raw[sensor]) if sensor in row_raw.index else np.nan
        ctx_pct = get_contextual_percentile(sensor, raw_val, hour_val)
        direction = "HAUSSE" if scores["signed"] > 0 else "BAISSE"
        row_out[f"top{rank}_capteur"]         = sensor
        row_out[f"top{rank}_label"]           = SENSOR_LABELS.get(sensor, sensor)
        row_out[f"top{rank}_shap_score"]      = round(scores["abs"], 6)
        row_out[f"top{rank}_direction"]       = direction
        row_out[f"top{rank}_valeur_brute"]    = round(raw_val, 2) if not np.isnan(raw_val) else ""
        row_out[f"top{rank}_percentile_ctx"]  = round(ctx_pct, 1)
        # Interpretation automatique simple
        if ctx_pct >= 90:
            qualif = "tres haute (top 10%)"
        elif ctx_pct >= 75:
            qualif = "haute (top 25%)"
        elif ctx_pct <= 10:
            qualif = "tres basse (bottom 10%)"
        elif ctx_pct <= 25:
            qualif = "basse (bottom 25%)"
        else:
            qualif = "dans la norme"
        row_out[f"top{rank}_interpretation"] = (
            f"{SENSOR_LABELS.get(sensor, sensor)} = {round(raw_val, 2) if not np.isnan(raw_val) else 'N/A'}"
            f" -> valeur {qualif} pour {hour_val}h ({direction} du score anomalie)"
        )

    rows.append(row_out)

df_shap = pd.DataFrame(rows)

path_shap = os.path.join(OUTPUT_DIR, "isolation_forest_v4_shap_analysis.csv")
df_shap.to_csv(path_shap, index=False)
print(f"  Exporte : {path_shap}")
print(f"  Lignes  : {len(df_shap)} anomalies mid+high analysees")
print(f"  Colonnes: {len(df_shap.columns)}")


# =============================================================================
# 12. RAPPORT FINAL
# =============================================================================
print(f"""
====================================================================
 ISOLATION FOREST v4 — RESULTATS FINAUX
====================================================================
 Dataset     : {DATA_PATH}
 Features    : {len(FEATURE_NAMES)} (drift-robust, aucune valeur brute)
 KS max      : {ks_s.max():.3f}  {'[OK]' if ks_s.max() < 0.30 else '[ATTENTION]'}

 Split 70/30 — TEST :
   normal: {(test_cat=='normal').sum():>5}
   low   : {(test_cat=='low').sum():>5}
   mid   : {(test_cat=='mid').sum():>5}
   high  : {(test_cat=='high').sum():>5}

 Test 100% dataset :
   normal: {(full_cat=='normal').sum():>6}
   low   : {(full_cat=='low').sum():>6}
   mid   : {(full_cat=='mid').sum():>6}
   high  : {(full_cat=='high').sum():>6}
   Agreement 70/30 vs 100% : {accord*100:.2f}%

 SHAP par capteur physique :
   Anomalies analysees : {len(df_shap)} (mid + high, jeu de test)
   Capteur le plus souvent responsable :
{
    df_shap["top1_capteur"].value_counts().head(3).to_string()
}

 Fichiers exportes :
   {path_full}
   {path_shap}
====================================================================
""")
