import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
DATA_PATH  = r"C:\Users\alexandre.batisse\.vscode\Projet\Projet_Stage_Scalian\data\Raw\energydata_complete.csv"
IF_CSV     = r"C:\Users\alexandre.batisse\.vscode\Projet\Projet_Stage_Scalian\outputs\output_Isolation_forest\isolation_forest_v4_scored_full.csv"
SVM_CSV    = r"C:\Users\alexandre.batisse\.vscode\Projet\Projet_Stage_Scalian\outputs\output_OCSVM\oneclass_svm_v1_scored_full.csv"
OUTPUT_DIR = r"C:\Users\alexandre.batisse\.vscode\Projet\Projet_Stage_Scalian\outputs\output_Autoencoder_dense"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_RATIO  = 0.70
WINDOWS      = [6, 12, 36, 72]    # cohérent avec IF v4
RANDOM_STATE = 42

# Seuils d'alerte (percentiles de la MSE train)
THR_LOW_Q  = 0.90
THR_MID_Q  = 0.95
THR_HIGH_Q = 0.99

# Hyperparamètres Autoencoder

HIDDEN_DIMS   = [64, 32]     # moins de capacité → moins de sur-apprentissage
ENCODING_DIM  = 16            # bottleneck plus serré → force la généralisation
DROPOUT_RATE  = 0.30          # dropout plus agressif pendant l'entraînement

LEARNING_RATE = 1e-3
EPOCHS        = 80          # max — early stopping actif
BATCH_SIZE    = 256
PATIENCE      = 10
VAL_SPLIT     = 0.15

# Device auto-detect
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Même mapping que IF v4 — bloc partagé
ENERGY_COLS = ["Appliances", "lights"]
TEMP_IN     = ["T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9"]
HUM_IN      = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_7", "RH_8", "RH_9"]
TEMP_OUT    = ["T6", "T_out"]
HUM_OUT     = ["RH_6", "RH_out"]
METEO       = ["Press_mm_hg", "Windspeed", "Tdewpoint"]
ALL_SENSORS = ENERGY_COLS + TEMP_IN + HUM_IN + TEMP_OUT + HUM_OUT + METEO

print("=" * 65)
print("  AUTOENCODER DENSE (PyTorch) — Anomalies énergétiques")
print("=" * 65)
print(f"  Device       : {DEVICE}")
print(f"  Architecture : [n_feat] -> {HIDDEN_DIMS} -> {ENCODING_DIM} (bottleneck) -> {list(reversed(HIDDEN_DIMS))} -> [n_feat]")
print(f"  Epochs max   : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  Patience : {PATIENCE}")
print()

# =============================================================================
# 1. CHARGEMENT — identique IF v4
# =============================================================================
print("=== 1. CHARGEMENT ===")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
df["hour"]       = df["date"].dt.hour
df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
print(f"  Dataset : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")
print(f"  Periode : {df['date'].min()} -> {df['date'].max()}")

# =============================================================================
# 2. FEATURE ENGINEERING — 7 familles drift-robustes (IDENTIQUE IF v4)
# =============================================================================
print("\n=== 2. FEATURE ENGINEERING ===")
feat = pd.DataFrame(index=df.index)

# FAMILLE 1 — Z-scores glissants
print("  [1/7] Z-scores glissants...")
for col in ALL_SENSORS:
    for w in WINDOWS:
        rm = df[col].rolling(w, min_periods=max(2, w // 2)).mean()
        rs = df[col].rolling(w, min_periods=max(2, w // 2)).std()
        feat[f"{col}_zscore_{w}"] = (df[col] - rm) / rs.replace(0, np.nan).fillna(1e-6)

# FAMILLE 2 — Deltas
print("  [2/7] Deltas...")
for col in ENERGY_COLS + TEMP_IN[:4] + HUM_IN[:4] + ["T_out", "RH_out"]:
    for lag in [1, 3, 6]:
        feat[f"{col}_delta_{lag}"] = df[col].diff(lag)

# FAMILLE 3 — tod_dev : reportée après split (section 4)
print("  [3/7] Deviations heure-du-jour -> reportees apres split (section 4)")

# FAMILLE 4 — Déviation weekend
print("  [4/7] Deviation weekend...")
for col in ENERGY_COLS:
    baseline_we = df.groupby("is_weekend")[col].transform("median")
    feat[f"{col}_we_dev"] = df[col] - baseline_we

# FAMILLE 5 — Rolling std
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

# FAMILLE 7 — Encodage cyclique
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
# 4. SPLIT TEMPOREL + FAMILLE 3 (tod_dev, fit train only)
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
N_FEATURES    = len(FEATURE_NAMES)
print(f"  Train : {split_idx:,} lignes  ({df_clean['date'].iloc[0].date()} -> {df_clean['date'].iloc[split_idx-1].date()})")
print(f"  Test  : {len(feat_clean)-split_idx:,} lignes  ({df_clean['date'].iloc[split_idx].date()} -> {df_clean['date'].iloc[-1].date()})")
print(f"  Features finales : {N_FEATURES}")

# =============================================================================
# 5. NORMALISATION SPÉCIFIQUE AE : StandardScaler -> MinMaxScaler [0,1]
# =============================================================================
# Pourquoi 2 étapes :
#   1) StandardScaler centre-réduit (robuste aux échelles très différentes entre
#      z-scores, deltas, température absolue centrée)
#   2) MinMaxScaler [0,1] pour matcher la Sigmoid finale du décodeur
#      (Sigmoid sort dans [0,1], donc si la cible n'est pas dans [0,1] la MSE
#      aura un plateau qu'aucun entraînement ne pourra franchir)
# Les deux scalers sont FIT sur train uniquement.
print("\n=== 5. NORMALISATION (StandardScaler + MinMaxScaler [0,1]) ===")
X_train = feat_clean.iloc[:split_idx].values.astype(np.float32)
X_test  = feat_clean.iloc[split_idx:].values.astype(np.float32)
X_full  = feat_clean.values.astype(np.float32)

std_scaler = StandardScaler()
X_train_s  = std_scaler.fit_transform(X_train)
X_test_s   = std_scaler.transform(X_test)
X_full_s   = std_scaler.transform(X_full)

mm_scaler  = MinMaxScaler(feature_range=(0, 1))
X_train_sc = mm_scaler.fit_transform(X_train_s).astype(np.float32)
X_test_sc  = mm_scaler.transform(X_test_s).astype(np.float32)
X_full_sc  = mm_scaler.transform(X_full_s).astype(np.float32)

print(f"  Scaler fitte sur {len(X_train):,} lignes")
print(f"  X_train range apres MinMax : [{X_train_sc.min():.3f}, {X_train_sc.max():.3f}]")


# =============================================================================
# 5bis. VERIFICATION DRIFT KS (feature-level train vs test)
# =============================================================================
from scipy.stats import ks_2samp

print("\n=== 5bis. VERIFICATION DRIFT KS ===")
ks_res = {}
for i, col in enumerate(FEATURE_NAMES):
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
# 6. ARCHITECTURE
# =============================================================================
print("\n=== 6. ARCHITECTURE ===")

class DenseAutoencoder(nn.Module):
    """
    Autoencoder dense symétrique.
    Encoder : [n_feat] -> 128 -> 64 -> 32
    Decoder : 32 -> 64 -> 128 -> [n_feat]  (Sigmoid en sortie)
    Chaque couche intermédiaire : Linear -> ELU -> BatchNorm -> Dropout.

    ELU plutôt que ReLU : gradient non nul pour entrées négatives, converge
    plus vite et reconstruit mieux les petites valeurs.
    """
    def __init__(self, n_features, hidden_dims, encoding_dim, dropout_rate):
        super().__init__()

        # Encoder
        enc_layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            enc_layers += [
                nn.Linear(in_dim, h_dim),
                nn.ELU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout_rate),
            ]
            in_dim = h_dim
        enc_layers += [nn.Linear(in_dim, encoding_dim), nn.ELU()]
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (miroir)
        dec_layers = []
        in_dim = encoding_dim
        for h_dim in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(in_dim, h_dim),
                nn.ELU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout_rate),
            ]
            in_dim = h_dim
        dec_layers += [nn.Linear(in_dim, n_features), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = DenseAutoencoder(N_FEATURES, HIDDEN_DIMS, ENCODING_DIM, DROPOUT_RATE).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Parametres totaux : {total_params:,}")

# =============================================================================
# 7. ENTRAÎNEMENT (fit sur train only, validation pour early stopping)
# =============================================================================
print(f"\n=== 7. ENTRAINEMENT (max {EPOCHS} epochs, patience={PATIENCE}) ===")

# Split validation (chronologique — on garde le plus récent du train pour valider)
n_val = int(len(X_train_sc) * VAL_SPLIT)
n_tr  = len(X_train_sc) - n_val
X_tr_t  = torch.tensor(X_train_sc[:n_tr])
X_val_t = torch.tensor(X_train_sc[n_tr:]).to(DEVICE)

train_loader = DataLoader(
    TensorDataset(X_tr_t, X_tr_t),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(len(X_tr_t) % BATCH_SIZE == 1),  # évite un batch de taille 1 (BatchNorm casse)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# FIX compatibilité PyTorch 2.4+ : plus de `verbose`, on log manuellement
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
)

train_losses, val_losses = [], []
best_val, best_weights = float("inf"), None
patience_cnt           = 0
prev_lr                = LEARNING_RATE

for epoch in range(1, EPOCHS + 1):
    # Phase train
    model.train()
    epoch_loss = 0.0
    n_seen     = 0
    for X_batch, _ in train_loader:
        X_batch = X_batch.to(DEVICE)
        optimizer.zero_grad()
        X_recon = model(X_batch)
        loss    = criterion(X_recon, X_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(X_batch)
        n_seen     += len(X_batch)
    epoch_loss /= max(n_seen, 1)

    # Phase val
    model.eval()
    with torch.no_grad():
        val_recon = model(X_val_t)
        val_loss  = criterion(val_recon, X_val_t).item()

    train_losses.append(epoch_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    # Log manuel du LR si le scheduler l'a réduit
    cur_lr = optimizer.param_groups[0]["lr"]
    lr_msg = ""
    if cur_lr < prev_lr:
        lr_msg = f"  [LR reduced to {cur_lr:.2e}]"
        prev_lr = cur_lr

    # Early stopping
    if val_loss < best_val - 1e-7:
        best_val     = val_loss
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1

    if epoch % 5 == 0 or patience_cnt == PATIENCE or epoch == 1 or lr_msg:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | train={epoch_loss:.6f} | val={val_loss:.6f} | lr={cur_lr:.2e} | patience={patience_cnt}/{PATIENCE}{lr_msg}")

    if patience_cnt >= PATIENCE:
        print(f"\n  Early stopping a l'epoch {epoch} (meilleure val_loss={best_val:.6f})")
        break

# Restaurer les meilleurs poids
model.load_state_dict(best_weights)
epochs_run = len(train_losses)
print(f"\n  Entrainement termine : {epochs_run} epochs | meilleure val_loss = {best_val:.6f}")

# =============================================================================
# 8. SCORING — MSE de reconstruction par observation
# =============================================================================
print("\n=== 8. SCORING ===")
model.eval()
with torch.no_grad():
    # Reconstruction en batches pour eviter de saturer la RAM GPU
    batch = 2048
    mse_all = np.empty(len(X_full_sc), dtype=np.float32)
    for i in range(0, len(X_full_sc), batch):
        X_batch_t = torch.tensor(X_full_sc[i:i+batch]).to(DEVICE)
        X_recon_t = model(X_batch_t)
        mse_all[i:i+batch] = torch.mean((X_batch_t - X_recon_t) ** 2, dim=1).cpu().numpy()

train_mask = np.arange(len(df_clean)) < split_idx
mse_train  = mse_all[train_mask]

# Seuils Q90/Q95/Q99 — cohérent avec IF v4 et OC-SVM
rolling_window_days = 28
cutoff_date = df_clean["date"].iloc[-1] - pd.Timedelta(days=rolling_window_days)
recent_mask = df_clean["date"] >= cutoff_date
mse_recent  = mse_all[recent_mask]
thr_low  = float(np.quantile(mse_recent, THR_LOW_Q))
thr_mid  = float(np.quantile(mse_recent, THR_MID_Q))
thr_high = float(np.quantile(mse_recent, THR_HIGH_Q))
print(f"  Seuils sur MSE train :")
print(f"    Q{int(THR_LOW_Q*100)}  (low)  = {thr_low:.6f}")
print(f"    Q{int(THR_MID_Q*100)}  (mid)  = {thr_mid:.6f}")
print(f"    Q{int(THR_HIGH_Q*100)}  (high) = {thr_high:.6f}")

def categorize(mse):
    cat = np.full(len(mse), "normal", dtype=object)
    cat[mse >= thr_low]  = "low"
    cat[mse >= thr_mid]  = "mid"
    cat[mse >= thr_high] = "high"
    return cat

# Severité = percentile rank sur la distribution train (même logique que IF v4)
train_sorted = np.sort(mse_train)
def severity(mse_vals):
    return np.searchsorted(train_sorted, mse_vals, side="right") / len(train_sorted)

full_cat = categorize(mse_all)
full_sev = severity(mse_all)

test_cat = full_cat[~train_mask]
print(f"\n  Distribution TEST :")
for cat, target in [("normal", 90), ("low", 5), ("mid", 4), ("high", 1)]:
    n   = (test_cat == cat).sum()
    pct = 100 * n / len(test_cat)
    flag = "[OK]" if abs(pct - target) < 5 else "[ATTENTION]"
    print(f"    {cat:<7}: {n:>5,}  ({pct:.2f}%)  cible ~{target}%  {flag}")

print(f"\n  Distribution GLOBALE (100%) :")
for cat in ["normal", "low", "mid", "high"]:
    n = (full_cat == cat).sum()
    pct = 100 * n / len(full_cat)
    print(f"    {cat:<7}: {n:>6,}  ({pct:.1f}%)")

# Validation temporelle — même garde-fou que tes autres scripts
print(f"\n  === VALIDATION DISTRIBUTION TEMPORELLE ===")
df_tmp = df_clean.copy()
df_tmp["anomaly_flag"]  = (full_cat != "normal").astype(int)
df_tmp["month_label"]   = df_tmp["date"].dt.strftime("%Y-%m")
monthly = df_tmp.groupby("month_label").agg(
    total=("anomaly_flag","count"),
    anomalies=("anomaly_flag","sum")
).reset_index()
monthly["taux_pct"] = (monthly["anomalies"] / monthly["total"] * 100).round(1)
print(monthly.to_string(index=False))
if monthly["taux_pct"].max() > 15:
    print("\n  [ATTENTION] Un mois depasse 15% -> drift residuel, revoir features.")
else:
    print("\n  [OK] Distribution temporelle homogene (< 15% pour tous les mois).")

# =============================================================================
# 9. EXPORT CSV — MÊME FORMAT que IF v4 + OC-SVM (pour merge 3 modèles)
# =============================================================================
print("\n=== 9. EXPORT CSV ===")

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

res_full = make_result_df(
    df_clean["date"].values, mse_all, full_sev, full_cat, df_clean, "full"
)
path_full = os.path.join(OUTPUT_DIR, "autoencoder_v1_scored_full.csv")
res_full.to_csv(path_full, index=False, encoding="utf-8-sig")
print(f"  Exporte : {path_full}")

path_anom = os.path.join(OUTPUT_DIR, "autoencoder_v1_anomalies.csv")
res_full[res_full["anomaly_category"].isin(["mid", "high"])].to_csv(
    path_anom, index=False, encoding="utf-8-sig"
)
n_anom = (res_full["anomaly_category"].isin(["mid", "high"])).sum()
print(f"  Exporte : {path_anom}  ({n_anom} anomalies mid+high)")

# =============================================================================
# 10. CONCORDANCE 3 MODÈLES (si IF et SVM dispo)
# =============================================================================
print("\n=== 10. CONCORDANCE 3 MODELES ===")
COMPARE3 = False
if os.path.exists(IF_CSV) and os.path.exists(SVM_CSV):
    print("  Fichiers IF et SVM detectes -> comparaison activee")
    df_if  = pd.read_csv(IF_CSV,  parse_dates=["date"])
    df_svm = pd.read_csv(SVM_CSV, parse_dates=["date"])

    df_3 = res_full[["date","Appliances","anomaly_flag","anomaly_category","raw_score"]].rename(
        columns={"anomaly_flag":"ae_flag","anomaly_category":"ae_cat","raw_score":"ae_score"}
    ).merge(
        df_if[["date","anomaly_flag","anomaly_category","raw_score"]].rename(
            columns={"anomaly_flag":"if_flag","anomaly_category":"if_cat","raw_score":"if_score"}),
        on="date", how="inner"
    ).merge(
        df_svm[["date","anomaly_flag","anomaly_category","raw_score"]].rename(
            columns={"anomaly_flag":"svm_flag","anomaly_category":"svm_cat","raw_score":"svm_score"}),
        on="date", how="inner"
    )

    n = len(df_3)
    all3    = ((df_3["ae_flag"]==1) & (df_3["if_flag"]==1) & (df_3["svm_flag"]==1)).sum()
    ae_svm  = ((df_3["ae_flag"]==1) & (df_3["if_flag"]==0) & (df_3["svm_flag"]==1)).sum()
    ae_if   = ((df_3["ae_flag"]==1) & (df_3["if_flag"]==1) & (df_3["svm_flag"]==0)).sum()
    if_svm  = ((df_3["ae_flag"]==0) & (df_3["if_flag"]==1) & (df_3["svm_flag"]==1)).sum()
    ae_only = ((df_3["ae_flag"]==1) & (df_3["if_flag"]==0) & (df_3["svm_flag"]==0)).sum()
    if_only = ((df_3["ae_flag"]==0) & (df_3["if_flag"]==1) & (df_3["svm_flag"]==0)).sum()
    sv_only = ((df_3["ae_flag"]==0) & (df_3["if_flag"]==0) & (df_3["svm_flag"]==1)).sum()
    none_   = ((df_3["ae_flag"]==0) & (df_3["if_flag"]==0) & (df_3["svm_flag"]==0)).sum()

    print(f"\n  ====== CONCORDANCE 3 MODELES ======")
    print(f"  Detectees par LES 3 MODELES    : {all3:5,}  ({all3/n*100:.1f}%)  <- PLUS FIABLES")
    print(f"  AE + OC-SVM  (pas IF)          : {ae_svm:5,}  ({ae_svm/n*100:.1f}%)")
    print(f"  AE + IF      (pas SVM)         : {ae_if:5,}  ({ae_if/n*100:.1f}%)")
    print(f"  IF + OC-SVM  (pas AE)          : {if_svm:5,}  ({if_svm/n*100:.1f}%)")
    print(f"  AE seulement                   : {ae_only:5,}  ({ae_only/n*100:.1f}%)")
    print(f"  IF seulement                   : {if_only:5,}  ({if_only/n*100:.1f}%)")
    print(f"  OC-SVM seulement               : {sv_only:5,}  ({sv_only/n*100:.1f}%)")
    print(f"  Normales pour les 3            : {none_:5,}  ({none_/n*100:.1f}%)")

    def concordance_label(row):
        active = []
        if row["ae_flag"]==1:  active.append("ae")
        if row["if_flag"]==1:  active.append("if")
        if row["svm_flag"]==1: active.append("svm")
        if len(active) == 3: return "all_three"
        if len(active) == 2: return "+".join(active)
        if len(active) == 1: return active[0]+"_only"
        return "normal"

    df_3["concordance"] = df_3.apply(concordance_label, axis=1)
    out_3 = os.path.join(OUTPUT_DIR, "comparison_three_models.csv")
    df_3.to_csv(out_3, index=False, encoding="utf-8-sig")
    print(f"\n  Comparatif 3 modeles exporte : {out_3}")
    COMPARE3 = True
else:
    missing = []
    if not os.path.exists(IF_CSV):  missing.append("IF")
    if not os.path.exists(SVM_CSV): missing.append("SVM")
    print(f"  (Fichiers manquants : {missing} -> mode comparaison desactive)")

# =============================================================================
# 11. FIGURES
# =============================================================================
print("\n=== 11. FIGURES ===")
colors_cat = {"low":"#2196F3", "mid":"#FF9800", "high":"#F44336"}

# FIGURE 1 — Courbe d'entraînement
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train_losses, label="Train MSE",      color="steelblue")
ax.plot(val_losses,   label="Validation MSE", color="coral", linestyle="--")
best_epoch = int(np.argmin(val_losses)) + 1
ax.axvline(best_epoch - 1, color="gray", linestyle=":", alpha=0.7,
           label=f"Best epoch ({best_epoch})")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title("Autoencoder Dense — Courbe d'entrainement")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ae_training_curve.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Figure 1 : ae_training_curve.png")

# FIGURE 2 — Timeline
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(res_full["date"], res_full["Appliances"], color="lightgray", lw=0.5, label="Appliances")
for cat, color in colors_cat.items():
    m = res_full["anomaly_category"] == cat
    ax.scatter(res_full.loc[m, "date"], res_full.loc[m, "Appliances"],
               color=color, s=10 if cat=="low" else 18,
               label=f"{cat} ({m.sum()})", zorder=3, alpha=0.8)
ax.set_xlabel("Date")
ax.set_ylabel("Appliances (Wh)")
ax.set_title("Autoencoder Dense — Timeline des anomalies")
ax.legend(loc="upper right")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ae_timeline.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Figure 2 : ae_timeline.png")

# FIGURE 3 — Distribution des MSE
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Autoencoder Dense — Distribution de la MSE de reconstruction", fontsize=13)

clip_max = float(np.percentile(mse_all, 99.5))
axes[0].hist(mse_all[train_mask],  bins=100, color="steelblue", alpha=0.7, label="Train")
axes[0].hist(mse_all[~train_mask], bins=100, color="coral",     alpha=0.5, label="Test")
axes[0].axvline(thr_low,  color="gold",   lw=1.5, linestyle="--", label=f"Q90={thr_low:.5f}")
axes[0].axvline(thr_mid,  color="orange", lw=1.5, linestyle="--", label=f"Q95={thr_mid:.5f}")
axes[0].axvline(thr_high, color="red",    lw=2,   linestyle="--", label=f"Q99={thr_high:.5f}")
axes[0].set_xlim(0, clip_max)
axes[0].set_xlabel("MSE de reconstruction")
axes[0].set_ylabel("Frequence")
axes[0].set_title("Histogramme MSE (train vs test)")
axes[0].legend(fontsize=8)

cat_counts = res_full["anomaly_category"].value_counts().reindex(["normal","low","mid","high"])
axes[1].bar(cat_counts.index, cat_counts.values,
            color=["#4CAF50","#2196F3","#FF9800","#F44336"])
axes[1].set_title("Repartition par categorie (100% dataset)")
axes[1].set_ylabel("Nombre d'observations")
for i, v in enumerate(cat_counts.values):
    axes[1].text(i, v + 10, f"{v}\n({v/len(res_full)*100:.1f}%)", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ae_score_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Figure 3 : ae_score_distribution.png")

# FIGURE 4 — Concordance 3 modèles
if COMPARE3:
    conc_colors = {
        "all_three": "#F44336",
        "ae+if":     "#FF5722",
        "ae+svm":    "#9C27B0",
        "if+svm":    "#FF9800",
        "ae_only":   "#2196F3",
        "if_only":   "#8BC34A",
        "svm_only":  "#00BCD4",
    }
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(res_full["date"], res_full["Appliances"], color="lightgray", lw=0.4, zorder=1)
    for conc, color in conc_colors.items():
        m = df_3["concordance"] == conc
        if m.sum() > 0:
            ax.scatter(df_3.loc[m,"date"], df_3.loc[m,"Appliances"],
                       s=12, color=color, alpha=0.85,
                       label=f"{conc} ({m.sum()})", zorder=3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliances (Wh)")
    ax.set_title("Concordance IF / OC-SVM / Autoencoder — Timeline")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ae_comparison_three_models.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Figure 4 : ae_comparison_three_models.png")

# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================
print(f"""
==================================================================
 AUTOENCODER DENSE — RESULTATS FINAUX
==================================================================
 Dataset     : {DATA_PATH}
 Features    : {N_FEATURES} (identique Isolation Forest v4)
 Device      : {DEVICE}
 Epochs      : {epochs_run} / {EPOCHS}  (meilleure val_loss = {best_val:.6f})

 Seuils MSE train :
   Q90 (low)  : {thr_low:.6f}
   Q95 (mid)  : {thr_mid:.6f}
   Q99 (high) : {thr_high:.6f}

 Distribution GLOBALE :
   normal : {(full_cat=='normal').sum():>6,}
   low    : {(full_cat=='low').sum():>6,}
   mid    : {(full_cat=='mid').sum():>6,}
   high   : {(full_cat=='high').sum():>6,}

 Fichiers exportes :
   {path_full}
   {path_anom}""" + (f"""
   {out_3}""" if COMPARE3 else "") + """
==================================================================
""")