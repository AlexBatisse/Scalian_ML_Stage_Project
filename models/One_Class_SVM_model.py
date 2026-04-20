import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
RAW_CSV    = r"C:\Users\alexandre.batisse\.vscode\Projet\Projet_Stage_Scalian\data\Raw\energydata_complete.csv"
IF_CSV     = r"C:\Users\alexandre.batisse\.vscode\Projet\Projet_Stage_Scalian\outputs\output_Isolation_forest\isolation_forest_v4_scored_full.csv"
OUTPUT_DIR = r"C:\Users\alexandre.batisse\.vscode\Projet\Projet_Stage_Scalian\outputs\output_OCSVM"
""

# Hyperparamètres One-Class SVM
# nu     : borne supérieure du taux d'anomalies attendu (équivalent "contamination" d'IF)
#           valeurs typiques : 0.03 à 0.10 selon le domaine
# kernel : "rbf" recommandé pour données continues multivariées
# gamma  : "scale" = 1 / (n_features * X.var()) — bonne valeur par défaut
#           augmenter gamma → frontière plus serrée (plus d'anomalies)
#           diminuer gamma  → frontière plus large (moins d'anomalies)
NU      = 0.02
KERNEL  = "rbf"
GAMMA   = "scale"

TRAIN_RATIO = 0.70
RANDOM_STATE = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("  ONE-CLASS SVM — Détection d'anomalies énergétiques")
print("=" * 65)
print(f"  nu={NU}  |  kernel={KERNEL}  |  gamma={GAMMA}")
print(f"  Split temporel : {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}")
print()

# =============================================================================
# 1. CHARGEMENT
# =============================================================================
print("[1/8] Chargement des données...")
df = pd.read_csv(RAW_CSV, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

df["hour"]        = df["date"].dt.hour
df["minute"]      = df["date"].dt.minute
df["day_of_week"] = df["date"].dt.dayofweek
df["month"]       = df["date"].dt.month
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)

print(f"  Dataset : {len(df):,} lignes x {len(df.columns)} colonnes")
print(f"  Période : {df['date'].min().date()} → {df['date'].max().date()}")

# =============================================================================
# 2. FEATURE ENGINEERING — 7 familles drift-robustes (IDENTIQUE IF v4)
# =============================================================================
print("\n[2/8] Feature engineering (aligne IF v4 pour comparaison propre)...")

ENERGY_COLS = ["Appliances", "lights"]
TEMP_IN     = ["T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9"]
HUM_IN      = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_7", "RH_8", "RH_9"]
TEMP_OUT    = ["T6", "T_out"]
HUM_OUT     = ["RH_6", "RH_out"]
METEO       = ["Press_mm_hg", "Windspeed", "Tdewpoint"]
ALL_SENSORS = ENERGY_COLS + TEMP_IN + HUM_IN + TEMP_OUT + HUM_OUT + METEO

feat = pd.DataFrame(index=df.index)

# FAMILLE 1 — Z-scores glissants (drift-robust par construction)
print("  [1/7] Z-scores glissants...")
for col in ALL_SENSORS:
    for w in [6, 12, 36, 72]:
        rm = df[col].rolling(w, min_periods=max(2, w // 2)).mean()
        rs = df[col].rolling(w, min_periods=max(2, w // 2)).std()
        feat[f"{col}_zscore_{w}"] = (df[col] - rm) / rs.replace(0, np.nan).fillna(1e-6)

# FAMILLE 2 — Deltas (taux de variation, independants du niveau absolu)
print("  [2/7] Deltas...")
for col in ENERGY_COLS + TEMP_IN[:4] + HUM_IN[:4] + ["T_out", "RH_out"]:
    for lag in [1, 3, 6]:
        feat[f"{col}_delta_{lag}"] = df[col].diff(lag)

# FAMILLE 3 — tod_dev : reportee apres split (section 4)
print("  [3/7] Deviations heure-du-jour -> reportees apres split")

# FAMILLE 4 — Deviation weekend
print("  [4/7] Deviation weekend...")
for col in ENERGY_COLS:
    baseline_we = df.groupby("is_weekend")[col].transform("median")
    feat[f"{col}_we_dev"] = df[col] - baseline_we

# FAMILLE 5 — Rolling std (variabilite locale)
print("  [5/7] Rolling std...")
for col in ENERGY_COLS + TEMP_IN[:3] + HUM_IN[:3]:
    for w in [6, 36]:
        feat[f"{col}_rstd_{w}"] = df[col].rolling(w, min_periods=2).std()

# FAMILLE 6 — Cross-capteurs
print("  [6/7] Cross-capteurs...")
feat["T_indoor_spread"]         = df[TEMP_IN].max(axis=1) - df[TEMP_IN].min(axis=1)
feat["T_indoor_outdoor_delta"]  = df[TEMP_IN].mean(axis=1) - df["T_out"]
feat["T6_T_out_delta"]          = df["T6"] - df["T_out"]
feat["RH_indoor_spread"]        = df[HUM_IN].max(axis=1) - df[HUM_IN].min(axis=1)
feat["RH_indoor_outdoor_delta"] = df[HUM_IN].mean(axis=1) - df["RH_out"]
feat["T1_dewpoint_margin"]      = df["T1"] - df["Tdewpoint"]
feat["energy_ratio"]            = df["lights"] / df["Appliances"].replace(0, np.nan).fillna(1)

# FAMILLE 7 — Encodage cyclique (periodique par construction)
print("  [7/7] Encodage cyclique...")
feat["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
feat["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
feat["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
feat["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

# Nettoyage NaN (warmup)
valid_mask = ~feat.isna().any(axis=1)
feat_clean = feat[valid_mask].reset_index(drop=True)
df         = df[valid_mask].reset_index(drop=True)

# Split temporel 70/30
n_train = int(len(feat_clean) * TRAIN_RATIO)
train_mask = np.zeros(len(feat_clean), dtype=bool)
train_mask[:n_train] = True
df["split"] = np.where(train_mask, "train", "test")

# FAMILLE 3 (tod_dev) — fit sur TRAIN uniquement, map sur tout
print("  Calcul des references heure-du-jour (sur train)...")
train_df = df.iloc[:n_train].copy()
for col in ENERGY_COLS:
    tod_med = train_df.groupby("hour")[col].median()
    tod_std = train_df.groupby("hour")[col].std().replace(0, 1e-6)
    bmed = df["hour"].map(tod_med)
    bstd = df["hour"].map(tod_std)
    feat_clean[f"{col}_tod_dev"]    = df[col].values - bmed.values
    feat_clean[f"{col}_tod_zscore"] = (df[col].values - bmed.values) / bstd.values

t_mean_train = train_df[TEMP_IN].mean(axis=1).mean()
feat_clean["T_indoor_mean_centered"] = df[TEMP_IN].mean(axis=1) - t_mean_train

print(f"  Lignes warmup supprimees : {(~valid_mask).sum()}")
print(f"  Features finales         : {feat_clean.shape[1]}")

# =============================================================================
# 3. SELECTION DES FEATURES (toutes les colonnes de feat_clean sont drift-robust)
# =============================================================================
FEATURE_COLS = list(feat_clean.columns)
print(f"\n[3/8] Features utilisees : {len(FEATURE_COLS)}")

X_train = feat_clean.iloc[:n_train].values
X_test  = feat_clean.iloc[n_train:].values
X_all   = feat_clean.values

# =============================================================================
# 3bis. VERIFICATION DRIFT KS (feature-level train vs test)
# =============================================================================
# Test de Kolmogorov-Smirnov pour chaque feature : mesure si la distribution
# train et la distribution test sont significativement differentes.
#   KS <= 0.30 : feature stable
#   0.30-0.50  : drift modere -> a surveiller
#   KS >  0.50 : drift severe -> la feature est sujette au biais saisonnier
from scipy.stats import ks_2samp

print("\n[3bis/8] Verification drift KS...")
ks_res = {}
for i, col in enumerate(FEATURE_COLS):
    ks_res[col] = ks_2samp(X_train[:, i], X_test[:, i])[0]
ks_s = pd.Series(ks_res).sort_values(ascending=False)

n_severe   = (ks_s > 0.5).sum()
n_moderate = ((ks_s > 0.3) & (ks_s <= 0.5)).sum()
n_stable   = (ks_s <= 0.3).sum()
print(f"  Severe   (KS > 0.5)  : {n_severe:>3}")
print(f"  Modere   (0.3-0.5)   : {n_moderate:>3}")
print(f"  Stable   (KS <= 0.3) : {n_stable:>3}")
print(f"  KS max               : {ks_s.max():.3f}  " +
      ("[OK]" if ks_s.max() < 0.30 else "[ATTENTION]"))

if n_severe > 0:
    print(f"\n  Top 5 features les plus drift-ees :")
    for feat_name, ks_val in ks_s.head(5).items():
        print(f"    {feat_name:<40} KS={ks_val:.3f}")

# =============================================================================
# 4. NORMALISATION
# =============================================================================
print("\n[4/8] Normalisation (StandardScaler)...")
scaler = RobustScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
X_all_sc   = scaler.transform(X_all)

# =============================================================================
# 5. ENTRAÎNEMENT ONE-CLASS SVM
# =============================================================================
print(f"\n[5/8] Entraînement One-Class SVM...")
print(f"  ATTENTION : peut prendre 3-10 min sur {n_train:,} obs. (normal pour OC-SVM)")


MAX_TRAIN_OCSVM = 5000
if len(X_train_sc) > MAX_TRAIN_OCSVM:
    # Stratification par heure du jour pour représenter tous les créneaux
    hours_train = df.loc[train_mask, "hour"].values
    idx_sub = []
    per_hour = MAX_TRAIN_OCSVM // 24
    for h in range(24):
        idx_h = np.where(hours_train == h)[0]
        chosen = np.random.choice(idx_h, min(per_hour, len(idx_h)), replace=False)
        idx_sub.extend(chosen.tolist())
    idx_sub = np.sort(idx_sub)
    X_fit = X_train_sc[idx_sub]
    print(f"  Sous-échantillonnage STRATIFIÉ par heure : {len(idx_sub):,} obs.")
else:
    X_fit = X_train_sc

ocsvm = OneClassSVM(nu=NU, kernel=KERNEL, gamma=GAMMA)
ocsvm.fit(X_fit)
print("  Entraînement terminé.")

# =============================================================================
# 6. SCORING
# =============================================================================
print("\n[6/8] Scoring du dataset complet...")

# decision_function retourne des scores signés :
#   > 0  → normal (dans la frontière)
#   < 0  → anomalie (hors de la frontière)
#   plus négatif → plus anormal
raw_scores = ocsvm.decision_function(X_all_sc)

# Inversion : score positif = plus anormal (cohérence avec IF)
anomaly_scores = -raw_scores

# Calcul des seuils sur les scores du train
train_scores = anomaly_scores[train_mask]
q90 = np.percentile(train_scores, 90)
q95 = np.percentile(train_scores, 95)
q99 = np.percentile(train_scores, 99)

print(f"  Seuils (sur train) :")
print(f"    Q90 = {q90:.4f}  → seuil 'low'")
print(f"    Q95 = {q95:.4f}  → seuil 'mid'")
print(f"    Q99 = {q99:.4f}  → seuil 'high'")

def classify(score):
    if score >= q99: return "high"
    if score >= q95: return "mid"
    if score >= q90: return "low"
    return "normal"

def severity_pct(score):
    return float(np.searchsorted(np.sort(train_scores), score) / len(train_scores) * 100)

df["raw_score"]        = anomaly_scores
df["severity_pct"]     = df["raw_score"].apply(severity_pct).round(2)
df["anomaly_category"] = df["raw_score"].apply(classify)
df["anomaly_flag"]     = (df["anomaly_category"].isin(["mid","high"])).astype(int)

dist = df["anomaly_category"].value_counts()
print(f"\n  Distribution finale :")
for cat in ["normal","low","mid","high"]:
    n = dist.get(cat, 0)
    print(f"    {cat:6s} : {n:5,}  ({n/len(df)*100:.1f}%)")


# VALIDATION : distribution mensuelle des anomalies
print("\n  === VALIDATION DISTRIBUTION TEMPORELLE ===")
df["month_label"] = df["date"].dt.strftime("%Y-%m")
monthly = df.groupby("month_label").agg(
    total=("anomaly_flag","count"),
    anomalies=("anomaly_flag","sum")
).reset_index()
monthly["taux_pct"] = (monthly["anomalies"] / monthly["total"] * 100).round(1)
print(monthly.to_string(index=False))
print("\n  Si un mois dépasse 15% → drift non corrigé, réduire nu ou retravailler les features.")

# =============================================================================
# 7. EXPORTS CSV
# =============================================================================
print("\n[7/8] Export des fichiers CSV...")

# 7.1 Dataset complet scoré
df_out = df[[
    "date","split","Appliances","lights",
    "T1","T2","T_out","RH_1","RH_out","Windspeed",
    "raw_score","severity_pct","anomaly_category","anomaly_flag"
]].copy()

out_full = os.path.join(OUTPUT_DIR, "oneclass_svm_v1_scored_full.csv")
df_out.to_csv(out_full, index=False, encoding="utf-8-sig")
print(f"  Exporté : {out_full}")

# 7.2 Anomalies mid+high uniquement (compatible moteur de règles)
out_shap = os.path.join(OUTPUT_DIR, "oneclass_svm_v1_anomalies.csv")
df_anom = df_out[df_out["anomaly_category"].isin(["mid","high"])].copy()
df_anom.to_csv(out_shap, index=False, encoding="utf-8-sig")
print(f"  Exporté : {out_shap}  ({len(df_anom)} anomalies)")

# =============================================================================
# 8. COMPARAISON AVEC ISOLATION FOREST (si fichier disponible)
# =============================================================================
print("\n[8/8] Visualisations...")

if os.path.exists(IF_CSV):
    print("  Fichier Isolation Forest détecté — comparaison activée")
    df_if = pd.read_csv(IF_CSV, parse_dates=["date"])
    df_if = df_if.rename(columns={
        "anomaly_category": "if_category",
        "anomaly_flag":     "if_flag",
        "raw_score":        "if_score",
        "severity_pct":     "if_severity"
    })
    df_cmp = df_out.merge(
        df_if[["date","if_category","if_flag","if_score","if_severity"]],
        on="date", how="inner"
    )
    df_cmp = df_cmp.rename(columns={
        "anomaly_category": "svm_category",
        "anomaly_flag":     "svm_flag",
        "raw_score":        "svm_score",
        "severity_pct":     "svm_severity"
    })

    # Concordance
    both   = ((df_cmp["svm_flag"]==1) & (df_cmp["if_flag"]==1)).sum()
    svm_only  = ((df_cmp["svm_flag"]==1) & (df_cmp["if_flag"]==0)).sum()
    if_only   = ((df_cmp["svm_flag"]==0) & (df_cmp["if_flag"]==1)).sum()
    neither   = ((df_cmp["svm_flag"]==0) & (df_cmp["if_flag"]==0)).sum()

    print(f"\n  ====== CONCORDANCE MODELES ======")
    print(f"  Détectées par LES DEUX     : {both:5,}  ({both/len(df_cmp)*100:.1f}%) ← anomalies les plus fiables")
    print(f"  Détectées OC-SVM seulement : {svm_only:5,}  ({svm_only/len(df_cmp)*100:.1f}%)")
    print(f"  Détectées IF seulement     : {if_only:5,}  ({if_only/len(df_cmp)*100:.1f}%)")
    print(f"  Normales pour les deux     : {neither:5,}  ({neither/len(df_cmp)*100:.1f}%)")

    df_cmp["concordance"] = "normal"
    df_cmp.loc[(df_cmp["svm_flag"]==1) & (df_cmp["if_flag"]==1), "concordance"] = "both"
    df_cmp.loc[(df_cmp["svm_flag"]==1) & (df_cmp["if_flag"]==0), "concordance"] = "svm_only"
    df_cmp.loc[(df_cmp["svm_flag"]==0) & (df_cmp["if_flag"]==1), "concordance"] = "if_only"

    out_cmp = os.path.join(OUTPUT_DIR, "comparison_if_vs_svm.csv")
    df_cmp.to_csv(out_cmp, index=False, encoding="utf-8-sig")
    print(f"\n  Tableau comparatif exporté : {out_cmp}")
    COMPARE_MODE = True
else:
    print("  (Fichier Isolation Forest non trouvé — mode comparaison désactivé)")
    COMPARE_MODE = False

# =============================================================================
# FIGURE 1 : Timeline OC-SVM
# =============================================================================
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(df["date"], df["Appliances"], color="lightgray", lw=0.5, label="Appliances")

colors_cat = {"low":"#2196F3", "mid":"#FF9800", "high":"#F44336"}
for cat, color in colors_cat.items():
    mask = df["anomaly_category"] == cat
    ax.scatter(df.loc[mask,"date"], df.loc[mask,"Appliances"],
               color=color, s=10 if cat=="low" else 18, label=cat, zorder=3, alpha=0.8)

ax.set_xlabel("Date")
ax.set_ylabel("Appliances (Wh)")
ax.set_title(f"One-Class SVM (nu={NU}) — Timeline des anomalies")
ax.legend(loc="upper right")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "svm_timeline.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Figure 1 : svm_timeline.png")

# =============================================================================
# FIGURE 2 : Distribution des scores
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("One-Class SVM — Distribution des scores d'anomalie", fontsize=13)

axes[0].hist(anomaly_scores[train_mask],  bins=100, color="steelblue", alpha=0.7, label="Train (normaux)")
axes[0].hist(anomaly_scores[~train_mask], bins=100, color="coral",     alpha=0.5, label="Test")
axes[0].axvline(q90, color="gold",   lw=1.5, linestyle="--", label=f"Q90={q90:.3f}")
axes[0].axvline(q95, color="orange", lw=1.5, linestyle="--", label=f"Q95={q95:.3f}")
axes[0].axvline(q99, color="red",    lw=2,   linestyle="--", label=f"Q99={q99:.3f}")
axes[0].set_xlabel("Score d'anomalie (plus élevé = plus anormal)")
axes[0].set_ylabel("Fréquence")
axes[0].set_title("Histogramme des scores")
axes[0].legend(fontsize=8)

cat_counts = df["anomaly_category"].value_counts().reindex(["normal","low","mid","high"])
colors_bar = ["#4CAF50","#2196F3","#FF9800","#F44336"]
axes[1].bar(cat_counts.index, cat_counts.values, color=colors_bar)
axes[1].set_title("Répartition par catégorie")
axes[1].set_ylabel("Nombre d'observations")
for i, v in enumerate(cat_counts.values):
    axes[1].text(i, v+10, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "svm_score_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Figure 2 : svm_score_distribution.png")

# =============================================================================
# FIGURE 3 : Scatter Appliances vs T_out
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
normal_mask = df["anomaly_category"] == "normal"
ax.scatter(df.loc[normal_mask,"T_out"], df.loc[normal_mask,"Appliances"],
           alpha=0.15, s=4, color="lightgray", label="Normal")
for cat, color in colors_cat.items():
    m = df["anomaly_category"] == cat
    ax.scatter(df.loc[m,"T_out"], df.loc[m,"Appliances"],
               s=20, color=color, alpha=0.7, label=cat, zorder=3)

ax.set_xlabel("T_out (°C)")
ax.set_ylabel("Appliances (Wh)")
ax.set_title("OC-SVM — Anomalies dans l'espace Appliances vs T_out")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "svm_scatter_appliances_tout.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Figure 3 : svm_scatter_appliances_tout.png")

# =============================================================================
# RÉSUMÉ CONSOLE
# =============================================================================
print("\n" + "=" * 65)
print("  RÉSUMÉ")
print("=" * 65)
print(f"  Observations totales  : {len(df):,}")
print(f"  Anomalies détectées   : {df['anomaly_flag'].sum():,}  ({df['anomaly_flag'].mean()*100:.1f}%)")
print(f"    - mid               : {(df['anomaly_category']=='mid').sum():,}")
print(f"    - high              : {(df['anomaly_category']=='high').sum():,}")
print(f"\n  Fichiers générés dans : {OUTPUT_DIR}")
print(f"    oneclass_svm_v1_scored_full.csv")
print(f"    oneclass_svm_v1_anomalies.csv")
if COMPARE_MODE:
    print(f"    comparison_if_vs_svm.csv")
print(f"    svm_timeline.png")
print(f"    svm_score_distribution.png")
print(f"    svm_scatter_appliances_tout.png")
if COMPARE_MODE:
    print(f"    comparison_if_vs_svm.png")
print("=" * 65)