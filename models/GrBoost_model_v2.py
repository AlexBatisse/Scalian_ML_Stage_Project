# ============================================================
# MODÈLE PRÉDICTIF DE CONSOMMATION ÉNERGÉTIQUE (IPMVP) - v3
# Baseline + Simulation What-if interactive
# Dataset : energydata_complete.csv (UCI - 4,5 mois, pas de 10 min)
# ============================================================
# Ce code illustre la MÉTHODOLOGIE IPMVP, pas des économies réalistes.
# ============================================================
# Principe IPMVP :
# Économies = Consommation de référence ajustée − Consommation prédite sous la mesure

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = 'C:\\Users\\alexandre.batisse\\.vscode\\Projet\\Projet_Stage_Scalian\\data\\Raw\\energydata_complete.csv'
OUTPUT_DIR = 'C:\\Users\\alexandre.batisse\\.vscode\\Projet\\Projet_Stage_Scalian\\outputs\\output_RG_Boost_model'
RANDOM_STATE = 42

# Ratio train/test : 0.30 = 30% pour entraîner, 70% pour simuler
# Plus le ratio est BAS, plus la période simulée est longue MAIS moins le
# modèle est entraîné. 0.30 est un compromis pour avoir ~100 jours de simu.
TRAIN_RATIO = 0.70

# ============================================================
# PARTIE 1 — CHARGEMENT ET PRÉTRAITEMENT
# ============================================================

def load_and_preprocess(path: str) -> pd.DataFrame:
    """Charge le CSV et applique le feature engineering.

    Règle d'or : AUCUNE feature ne doit être construite à partir de la
    valeur actuelle de la cible (Appliances à l'instant t). Les lags
    (Appliances à t-1, t-2...) sont autorisés car c'est du passé connu.
    """
    print(f"\n{'='*60}\nCHARGEMENT DU DATASET\n{'='*60}")
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"Période        : {df['date'].min()} → {df['date'].max()}")
    print(f"Observations   : {len(df):,} (pas de 10 min)")
    print(f"Durée totale   : {(df['date'].max() - df['date'].min()).days} jours")

    # --- 1.1 Colonnes inutiles ---
    df.drop(columns=['rv1', 'rv2', 'Visibility'], inplace=True)

    # --- 1.2 Winsorisation des outliers sur Appliances ---
    p99 = df['Appliances'].quantile(0.99)
    n_out = (df['Appliances'] > p99).sum()
    df['Appliances'] = df['Appliances'].clip(upper=p99)
    print(f"Outliers traités : {n_out} valeurs > {p99:.0f} Wh ramenées au P99")

    # --- 1.3 Features temporelles ---
    df['hour']        = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month']       = df['date'].dt.month
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['is_night']    = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
    df['is_workhour'] = ((df['hour'] >= 8) & (df['hour'] < 18) &
                         (df['day_of_week'] < 5)).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # --- 1.4 Features physiques dérivées (AUCUNE ne dépend de la cible) ---
    df['T_indoor_avg']           = df[[f'T{i}' for i in range(1, 10)]].mean(axis=1)
    df['T_indoor_outdoor_delta'] = df['T_indoor_avg'] - df['T_out']
    df['RH_indoor_avg']          = df[[f'RH_{i}' for i in range(1, 10)]].mean(axis=1)

    for col in ['T1', 'T2', 'T3']:
        df[f'{col}_rstd_6'] = df[col].rolling(6, min_periods=1).std().fillna(0)

    # --- 1.5 Lags de la cible (passé connu, pas de fuite) ---
    for lag in [1, 2, 3, 6, 144]:
        df[f'Appliances_lag_{lag}'] = df['Appliances'].shift(lag)
    df['Appliances_ma_6']   = df['Appliances'].shift(1).rolling(6,   min_periods=1).mean()
    df['Appliances_ma_144'] = df['Appliances'].shift(1).rolling(144, min_periods=1).mean()

    n_before = len(df)
    df.dropna(inplace=True)
    print(f"NaN supprimés (lags) : {n_before - len(df)}")
    print(f"Observations finales : {len(df):,}")
    return df


# ============================================================
# PARTIE 2 — SPLIT TEMPOREL ET ENTRAÎNEMENT
# ============================================================

FEATURES = [
    'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Tdewpoint',
    'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9',
    'RH_1', 'RH_2', 'RH_3', 'RH_4', 'RH_5', 'RH_6', 'RH_7', 'RH_8', 'RH_9',
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_night', 'is_workhour',
    'hour_sin', 'hour_cos',
    'T_indoor_avg', 'T_indoor_outdoor_delta', 'RH_indoor_avg',
    'T1_rstd_6', 'T2_rstd_6', 'T3_rstd_6',
    'lights',
    'Appliances_lag_1', 'Appliances_lag_2', 'Appliances_lag_3',
    'Appliances_lag_6', 'Appliances_lag_144',
    'Appliances_ma_6', 'Appliances_ma_144',
]
TARGET = 'Appliances'
T_COLS = [f'T{i}' for i in range(1, 10)]


def split_and_train(df: pd.DataFrame) -> dict:
    print(f"\n{'='*60}\nENTRAÎNEMENT DU MODÈLE (train={TRAIN_RATIO:.0%}, test={1-TRAIN_RATIO:.0%})\n{'='*60}")

    split_idx = int(len(df) * TRAIN_RATIO)
    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()

    d_tr = (df_train['date'].max() - df_train['date'].min()).days
    d_te = (df_test ['date'].max() - df_test ['date'].min()).days
    print(f"Train : {df_train['date'].min().date()} → {df_train['date'].max().date()} "
          f"({len(df_train):,} obs ≈ {d_tr} j)")
    print(f"Simu  : {df_test ['date'].min().date()} → {df_test ['date'].max().date()} "
          f"({len(df_test ):,} obs ≈ {d_te} j)")

    X_train, y_train = df_train[FEATURES], df_train[TARGET]
    X_test,  y_test  = df_test [FEATURES], df_test [TARGET]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='rmse',
        early_stopping_rounds=30,
    )
    print("\nEntraînement en cours (XGBoost)...")
    model.fit(X_train_sc, y_train, eval_set=[(X_test_sc, y_test)], verbose=False)

    y_pred = model.predict(X_test_sc)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n--- Performances sur la période de simulation ---")
    print(f"MAE  : {mae:6.2f} Wh")
    print(f"RMSE : {rmse:6.2f} Wh")
    print(f"R²   : {r2:6.4f}")

    # Mémoriser les plages observées en entraînement pour les warnings
    feature_ranges = {
        col: (X_train[col].min(), X_train[col].max()) for col in FEATURES
    }

    return {
        'model': model, 'scaler': scaler,
        'df_train': df_train, 'df_test': df_test,
        'X_train': X_train, 'X_test': X_test,
        'y_test': y_test, 'y_pred': y_pred,
        'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2},
        'feature_ranges': feature_ranges,
    }


# ============================================================
# PARTIE 3 — SCÉNARIOS WHAT-IF
# ============================================================
#
# COMMENT CHAQUE SCÉNARIO CALCULE LES NOUVELLES VALEURS :
#
# 1. On part de X_test (les features RÉELLES observées pendant la période
#    de simulation, instant par instant, tous les 10 minutes).
# 2. On modifie certaines COLONNES pour refléter la mesure d'efficacité :
#    - Scénario A (chauffage -1°C)  : on soustrait 1°C aux colonnes T1..T9
#    - Scénario B (extinction nuit) : on force T1..T9 à une valeur basse
#      entre 22h et 6h
#    - Scénario C (éclairage -50%)  : on multiplie 'lights' par 0.5
# 3. On recalcule les features dérivées (T_indoor_avg, etc.) pour rester
#    cohérent (sinon T_indoor_avg ne correspondrait plus à la moyenne
#    des T1..T9 modifiées).
# 4. On passe ce X_modified dans le modèle → il prédit ce que 'Appliances'
#    SERAIT dans ces conditions modifiées.
# 5. Différence baseline − modifié = économies.
#
# ⚠️ Le modèle ne "comprend" pas physiquement la mesure. Il applique les
#    corrélations qu'il a apprises. Si 'Appliances' ne dépend pas
#    causalement de la température intérieure (cas de ce dataset), le
#    résultat sera peu fiable voire contre-intuitif.
# ============================================================

def recompute_derived(X: pd.DataFrame) -> pd.DataFrame:
    X['T_indoor_avg']           = X[T_COLS].mean(axis=1)
    X['T_indoor_outdoor_delta'] = X['T_indoor_avg'] - X['T_out']
    return X


def scenario_A_heating_setpoint(X_test: pd.DataFrame, delta: float = 1.0) -> pd.DataFrame:
    """A : -1°C sur la consigne en journée (6h-22h).
    Variation modérée, reste généralement dans le domaine d'entraînement."""
    X = X_test.copy()
    mask = X['is_night'] == 0
    X.loc[mask, T_COLS] = X.loc[mask, T_COLS] - delta
    return recompute_derived(X)


def scenario_B_night_setback(X_test: pd.DataFrame, setback: float = 2.0) -> pd.DataFrame:
    """B : abaissement nocturne modéré (22h-6h).
    On baisse les T° intérieures de `setback` °C la nuit (au lieu de
    forcer T_out+4 qui sortait du domaine). C'est plus réaliste : un
    thermostat programmable typique baisse de 2-3°C la nuit."""
    X = X_test.copy()
    mask = X['is_night'] == 1
    X.loc[mask, T_COLS] = X.loc[mask, T_COLS] - setback
    return recompute_derived(X)


def scenario_C_lighting(X_test: pd.DataFrame, reduction_pct: float = 0.50) -> pd.DataFrame:
    """C : réduction éclairage (LED / détecteurs).
    Action DIRECTE sur une feature d'entrée → résultat plus fiable."""
    X = X_test.copy()
    X['lights'] = X['lights'] * (1 - reduction_pct)
    return X


def scenario_D_night_off_appliances(X_test: pd.DataFrame) -> pd.DataFrame:
    """D : extinction des veilles électriques la nuit.
    On simule : les appareils en veille sont coupés la nuit. On agit via
    les lags passés (on les diminue artificiellement la nuit) pour
    refléter un comportement persistant. Plus direct que la température.
    """
    X = X_test.copy()
    mask = X['is_night'] == 1
    lag_cols = [c for c in X.columns if 'Appliances_lag_' in c or 'Appliances_ma_' in c]
    # On diminue de 15% les lags la nuit (hypothèse : coupure des veilles)
    X.loc[mask, lag_cols] = X.loc[mask, lag_cols] * 0.85
    # L'éclairage résiduel s'éteint aussi
    X.loc[mask, 'lights'] = X.loc[mask, 'lights'] * 0.3
    return X


def scenario_E_combined(X_test: pd.DataFrame) -> pd.DataFrame:
    """E : bouquet combiné (A + B + C)."""
    X = scenario_A_heating_setpoint(X_test, delta=1.0)
    X = scenario_B_night_setback(X, setback=2.0)
    X = scenario_C_lighting(X, reduction_pct=0.30)
    return X


SCENARIOS = {
    '1': {
        'name': "Baisse consigne chauffage -1°C en journée",
        'description': ("Modifie T1..T9 de -1°C entre 6h et 22h.\n"
                        "   ⚠️ Appliances ne mesure PAS le chauffage dans ce dataset,\n"
                        "      le résultat reflète des corrélations indirectes."),
        'func': lambda X: scenario_A_heating_setpoint(X, delta=1.0),
    },
    '2': {
        'name': "Abaissement nocturne chauffage -2°C",
        'description': ("Modifie T1..T9 de -2°C entre 22h et 6h (thermostat\n"
                        "   programmable). Reste dans le domaine d'entraînement."),
        'func': lambda X: scenario_B_night_setback(X, setback=2.0),
    },
    '3': {
        'name': "Réduction éclairage -50%",
        'description': ("Multiplie 'lights' par 0.5 (remplacement LED ou\n"
                        "   détecteurs). Action DIRECTE sur une feature d'entrée."),
        'func': lambda X: scenario_C_lighting(X, reduction_pct=0.50),
    },
    '4': {
        'name': "Coupure des veilles électriques la nuit",
        'description': ("Réduit les lags de Appliances de 15% la nuit (simule\n"
                        "   une coupure automatique des appareils en veille)."),
        'func': scenario_D_night_off_appliances,
    },
    '5': {
        'name': "Bouquet combiné (1 + 2 + 3)",
        'description': "Application simultanée des scénarios 1, 2 et 3.",
        'func': scenario_E_combined,
    },
}


# ============================================================
# PARTIE 4 — SIMULATION ET RAPPORT
# ============================================================

def check_extrapolation(X_modified: pd.DataFrame, feature_ranges: dict,
                        tolerance: float = 0.05) -> list:
    """Vérifie que les features modifiées restent dans le domaine
    d'entraînement. Retourne une liste de features problématiques."""
    warnings_list = []
    for col in FEATURES:
        train_min, train_max = feature_ranges[col]
        train_range = train_max - train_min
        if train_range == 0:
            continue
        # On accepte une marge de `tolerance` × la plage d'entraînement
        lo = train_min - tolerance * train_range
        hi = train_max + tolerance * train_range
        n_out = ((X_modified[col] < lo) | (X_modified[col] > hi)).sum()
        if n_out > 0:
            pct = n_out / len(X_modified) * 100
            warnings_list.append(
                f"  • {col:25s} : {n_out:5d} valeurs ({pct:.1f}%) hors plage "
                f"d'entraînement [{train_min:.2f}, {train_max:.2f}]"
            )
    return warnings_list


def run_scenario(scenario_key: str, artifacts: dict) -> dict:
    scenario = SCENARIOS[scenario_key]
    model, scaler = artifacts['model'], artifacts['scaler']
    X_test, y_pred_baseline = artifacts['X_test'], artifacts['y_pred']

    # Application de la mesure
    X_modified = scenario['func'](X_test)

    # Vérification d'extrapolation
    extrap_warnings = check_extrapolation(X_modified, artifacts['feature_ranges'])

    X_modified_sc = scaler.transform(X_modified[FEATURES])
    y_pred_modified = model.predict(X_modified_sc)

    # Formule IPMVP
    savings_per_obs = y_pred_baseline - y_pred_modified

    total_baseline_kwh = y_pred_baseline.sum() / 1000
    total_scenario_kwh = y_pred_modified.sum() / 1000
    total_savings_kwh  = savings_per_obs.sum() / 1000
    pct_savings        = (savings_per_obs.sum() / y_pred_baseline.sum()) * 100

    n_days = max((artifacts['df_test']['date'].max()
                  - artifacts['df_test']['date'].min()).days, 1)
    annual_savings_kwh = total_savings_kwh * (365 / n_days)

    return {
        'scenario': scenario,
        'y_pred_baseline': y_pred_baseline,
        'y_pred_modified': y_pred_modified,
        'savings_per_obs': savings_per_obs,
        'total_baseline_kwh': total_baseline_kwh,
        'total_scenario_kwh': total_scenario_kwh,
        'total_savings_kwh':  total_savings_kwh,
        'pct_savings':        pct_savings,
        'annual_savings_kwh': annual_savings_kwh,
        'n_days':             n_days,
        'extrap_warnings':    extrap_warnings,
    }


def print_report(result: dict):
    s = result['scenario']
    print(f"\n{'='*60}\nRÉSULTATS — {s['name']}\n{'='*60}")
    print(f"{s['description']}\n")
    print(f"Période simulée         : {result['n_days']} jours")
    print(f"Baseline (référence)    : {result['total_baseline_kwh']:8.1f} kWh")
    print(f"Après mesure (prédit)   : {result['total_scenario_kwh']:8.1f} kWh")
    print(f"{'-'*40}")
    print(f"ÉCONOMIES ESTIMÉES      : {result['total_savings_kwh']:8.1f} kWh "
          f"({result['pct_savings']:+.1f}%)")
    print(f"Extrapolation annuelle  : {result['annual_savings_kwh']:8.1f} kWh/an")

    if result['extrap_warnings']:
        print(f"\n⚠️  ATTENTION : extrapolation hors du domaine d'entraînement")
        print("   Le modèle n'a jamais vu ces valeurs pendant l'entraînement,")
        print("   ses prédictions sont donc peu fiables :")
        for w in result['extrap_warnings'][:5]:  # max 5 warnings
            print(w)
        if len(result['extrap_warnings']) > 5:
            print(f"   ... et {len(result['extrap_warnings'])-5} autres features")

    if result['total_savings_kwh'] < 0:
        print("\n⚠️  Économie NÉGATIVE : la mesure augmente la consommation prédite.")
        print("    Causes possibles :")
        print("    - Corrélations non causales apprises par le modèle")
        print("    - Extrapolation (voir warnings ci-dessus)")
        print("    - La variable cible ne dépend pas causalement de la mesure")


# ============================================================
# PARTIE 5 — VISUALISATIONS
# ============================================================

def plot_baseline(artifacts: dict):
    fig, ax = plt.subplots(figsize=(14, 5))
    dates = artifacts['df_test']['date'].values
    ax.plot(dates, artifacts['y_test'].values, label='Consommation reelle',
            alpha=0.6, color='steelblue', linewidth=0.7)
    ax.plot(dates, artifacts['y_pred'], label='Baseline predite (modele)',
            alpha=0.85, color='orange', linewidth=0.7)
    m = artifacts['metrics']
    n_days = (artifacts['df_test']['date'].max()
              - artifacts['df_test']['date'].min()).days
    ax.set_title(f"Baseline IPMVP sur {n_days} jours - R2={m['r2']:.3f} | "
                 f"MAE={m['mae']:.1f} Wh | RMSE={m['rmse']:.1f} Wh")
    ax.set_ylabel('Consommation (Wh)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'baseline_vs_real.png')
    plt.savefig(path, dpi=130); print(f"-> {path}")
    plt.show()



def diagnostic_model(artifacts: dict):
    """Vérifications complémentaires pour valider la qualité du modèle.

    Un modèle de prédiction énergétique n'est utilisable pour de l'IPMVP
    que si :
    - Il n'a pas de biais systématique (erreur moyenne proche de 0)
    - Les erreurs sont distribuées de façon ~normale
    - La qualité est stable selon les périodes (jour/nuit, semaine/weekend)
    """
    y_test = artifacts['y_test'].values
    y_pred = artifacts['y_pred']
    df_test = artifacts['df_test']
    residuals = y_test - y_pred

    print(f"\n{'='*60}\nDIAGNOSTIC DU MODÈLE\n{'='*60}")

    # --- 1. Biais global ---
    bias = residuals.mean()
    print(f"\n1. Biais systématique")
    print(f"   Erreur moyenne (réel - prédit) : {bias:+.2f} Wh")
    if abs(bias) < 5:
        print(f"   → OK, biais négligeable (< 5 Wh)")
    else:
        print(f"   → Attention, biais notable : le modèle "
              f"{'sous-' if bias > 0 else 'sur-'}estime en moyenne")

    # --- 2. Distribution des résidus ---
    print(f"\n2. Distribution des résidus")
    print(f"   Médiane     : {np.median(residuals):+.2f} Wh")
    print(f"   Écart-type  : {residuals.std():.2f} Wh")
    print(f"   Percentiles : P5={np.percentile(residuals, 5):+.1f} | "
          f"P95={np.percentile(residuals, 95):+.1f} Wh")

    # --- 3. Performance par période ---
    print(f"\n3. Performance par période")
    df_diag = df_test.copy()
    df_diag['y_true'] = y_test
    df_diag['y_pred'] = y_pred
    df_diag['err_abs'] = np.abs(residuals)

    periods = {
        'Journée (6h-22h)'  : df_diag['is_night'] == 0,
        'Nuit (22h-6h)'     : df_diag['is_night'] == 1,
        'Semaine (lun-ven)' : df_diag['is_weekend'] == 0,
        'Weekend'           : df_diag['is_weekend'] == 1,
    }
    for label, mask in periods.items():
        if mask.sum() == 0:
            continue
        mae_p = df_diag.loc[mask, 'err_abs'].mean()
        mean_p = df_diag.loc[mask, 'y_true'].mean()
        pct = mae_p / mean_p * 100 if mean_p > 0 else 0
        print(f"   {label:20s} : MAE={mae_p:5.1f} Wh  "
              f"(conso moy. {mean_p:5.1f} Wh → erreur relative {pct:.0f}%)")

    # --- 4. Graphiques de diagnostic ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 4a. Scatter prédit vs réel
    ax = axes[0, 0]
    ax.scatter(y_test, y_pred, alpha=0.2, s=5, color='steelblue')
    lim = [0, max(y_test.max(), y_pred.max())]
    ax.plot(lim, lim, 'r--', linewidth=1.5, label='Diagonale parfaite (y=x)')
    ax.set_xlabel('Consommation réelle (Wh)')
    ax.set_ylabel('Consommation prédite (Wh)')
    ax.set_title('Prédit vs. Réel\n(plus les points sont proches de la diagonale, mieux c\'est)')
    ax.legend(); ax.grid(alpha=0.3)

    # 4b. Histogramme des résidus
    ax = axes[0, 1]
    ax.hist(residuals, bins=60, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zéro (idéal)')
    ax.axvline(bias, color='orange', linestyle='-', linewidth=1.5,
               label=f'Biais = {bias:+.1f} Wh')
    ax.set_xlabel('Résidu = réel - prédit (Wh)')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution des erreurs\n(devrait être centrée sur 0 et ~symétrique)')
    ax.legend(); ax.grid(alpha=0.3)

    # 4c. Résidus dans le temps (détection de dérives)
    ax = axes[1, 0]
    dates = df_test['date'].values
    # Moyenne glissante sur 1 jour (144 points à 10 min)
    rolling_res = pd.Series(residuals).rolling(144, min_periods=1).mean()
    ax.plot(dates, rolling_res, color='crimson', linewidth=1.2)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.fill_between(dates, rolling_res, 0,
                    where=(rolling_res > 0), alpha=0.3, color='crimson',
                    label='Sous-estimation')
    ax.fill_between(dates, rolling_res, 0,
                    where=(rolling_res < 0), alpha=0.3, color='seagreen',
                    label='Sur-estimation')
    ax.set_xlabel('Date')
    ax.set_ylabel('Résidu moyen sur 24h (Wh)')
    ax.set_title('Dérive temporelle\n(pas de tendance systématique = modèle stable)')
    ax.legend(); ax.grid(alpha=0.3)

    # 4d. Résidus vs. niveau de consommation
    ax = axes[1, 1]
    ax.scatter(y_pred, residuals, alpha=0.2, s=5, color='steelblue')
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Consommation prédite (Wh)')
    ax.set_ylabel('Résidu = réel - prédit (Wh)')
    ax.set_title('Résidus vs. Prédiction\n(nuage homogène = OK ; forme en entonnoir = problème)')
    ax.grid(alpha=0.3)

    plt.suptitle(f"Diagnostic du modèle — R²={artifacts['metrics']['r2']:.3f} | "
                 f"MAE={artifacts['metrics']['mae']:.1f} Wh",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'model_diagnostic.png')
    plt.savefig(path, dpi=130); print(f"\n-> {path}")
    plt.show()

    # --- 5. Verdict ---
    print(f"\n5. Verdict")
    r2 = artifacts['metrics']['r2']
    mean_conso = y_test.mean()
    rel_mae = artifacts['metrics']['mae'] / mean_conso * 100
    if r2 > 0.7 and rel_mae < 25:
        verdict = "✅ Modèle fiable pour l'IPMVP"
    elif r2 > 0.4 and rel_mae < 45:
        verdict = "⚠️  Modèle utilisable avec prudence (marge d'erreur ~ taille des économies)"
    else:
        verdict = "❌ Modèle insuffisant pour mesurer des économies fiables"
    print(f"   R² = {r2:.3f}  |  MAE relative = {rel_mae:.1f}%")
    print(f"   {verdict}")


def plot_scenario(result: dict, artifacts: dict):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    dates = artifacts['df_test']['date'].values

    ax1 = axes[0]
    ax1.plot(dates, result['y_pred_baseline'], label='Baseline (sans mesure)',
             color='steelblue', linewidth=0.7, alpha=0.8)
    ax1.plot(dates, result['y_pred_modified'], label='Avec mesure appliquee',
             color='seagreen', linewidth=0.7, alpha=0.8)
    ax1.fill_between(dates, result['y_pred_baseline'], result['y_pred_modified'],
                     where=(result['y_pred_baseline'] > result['y_pred_modified']),
                     alpha=0.25, color='seagreen', label='Economies')
    ax1.fill_between(dates, result['y_pred_baseline'], result['y_pred_modified'],
                     where=(result['y_pred_baseline'] < result['y_pred_modified']),
                     alpha=0.25, color='tomato', label='Sur-consommation')
    ax1.set_title(f"Scenario : {result['scenario']['name']} "
                  f"(sur {result['n_days']} jours)")
    ax1.set_ylabel('Consommation (Wh)')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2 = axes[1]
    labels = ['Baseline', 'Avec mesure', 'Economies']
    values = [result['total_baseline_kwh'],
              result['total_scenario_kwh'],
              result['total_savings_kwh']]
    colors = ['steelblue', 'seagreen', 'tomato' if values[2] > 0 else 'firebrick']
    bars = ax2.bar(labels, values, color=colors, alpha=0.85)
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{v:.0f} kWh', ha='center', va='bottom', fontweight='bold')
    ax2.set_ylabel('Energie (kWh)')
    ax2.set_title(f"Bilan sur {result['n_days']} jours - {result['pct_savings']:+.1f}% | "
                  f"{result['annual_savings_kwh']:.0f} kWh/an extrapole")
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    safe_name = result['scenario']['name'][:30].replace(' ', '_').replace('/', '_')
    path = os.path.join(OUTPUT_DIR, f"scenario_{safe_name}.png")
    plt.savefig(path, dpi=130); print(f"-> {path}")
    plt.show()


# ============================================================
# PARTIE 6 — EXPORT CSV
# ============================================================

def export_results(result: dict, artifacts: dict):
    df_out = artifacts['df_test'][['date']].copy()
    df_out['consommation_reelle']    = artifacts['y_test'].values
    df_out['baseline_predite_Wh']    = artifacts['y_pred']
    df_out['apres_mesure_predit_Wh'] = result['y_pred_modified']
    df_out['economie_Wh']            = result['savings_per_obs']
    path = os.path.join(OUTPUT_DIR, 'whatif_results.csv')
    df_out.to_csv(path, index=False)
    print(f"-> {path}")


# ============================================================
# PARTIE 7 — MENU INTERACTIF
# ============================================================

def interactive_menu(artifacts: dict):
    while True:
        print(f"\n{'='*60}")
        print("SIMULATION WHAT-IF - Quelle mesure voulez-vous tester ?")
        print('='*60)
        for key, sc in SCENARIOS.items():
            print(f"  [{key}] {sc['name']}")
        print(f"  [q] Quitter")

        choice = input("\nVotre choix : ").strip().lower()
        if choice in ('q', 'quit', 'exit', ''):
            print("Fin de la session.")
            break
        if choice not in SCENARIOS:
            print("Choix invalide, reessayez.")
            continue

        result = run_scenario(choice, artifacts)
        print_report(result)
        plot_scenario(result, artifacts)
        export_results(result, artifacts)

        again = input("\nTester un autre scenario ? (o/n) : ").strip().lower()
        if again not in ('o', 'oui', 'y', 'yes'):
            print("Fin de la session.")
            break


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():
    df = load_and_preprocess(DATA_PATH)
    artifacts = split_and_train(df)
    plot_baseline(artifacts)
    diagnostic_model(artifacts)
    interactive_menu(artifacts)


if __name__ == '__main__':
    main()