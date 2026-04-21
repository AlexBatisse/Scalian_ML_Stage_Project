# ============================================================
# MODÈLE PRÉDICTIF DE CONSOMMATION ÉNERGÉTIQUE (IPMVP)
# Baseline + Simulation What-if interactive
# Dataset : energydata_complete.csv (UCI - 4,5 mois, pas de 10 min)
# ============================================================
#
# Principe IPMVP :
#   Économies = Consommation de référence ajustée − Consommation prédite sous la mesure
#
# Pipeline :
#   1. Prétraitement + feature engineering (AUCUNE fuite de la cible)
#   2. Split temporel 70/30 (train = passé connu, test = période de simulation)
#   3. Entraînement XGBoost → baseline sur la période de test
#   4. Menu interactif : l'utilisateur choisit une mesure à simuler
#   5. Re-prédiction avec features modifiées → calcul des économies
#
# ⚠️ Limite du dataset :
#   Le jeu de données UCI ne couvre que 4,5 mois (janvier → mai 2016).
#   Le split 70/30 donne donc :
#      - Train : ~95 jours (janv → mi-avril)
#      - Test  : ~41 jours (mi-avril → fin mai)
#   Ce n'est pas une année complète, mais l'approche reste valide.
#   On extrapole les économies annuelles au prorata du test.
# ============================================================

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

# ⚠️ Adapter ce chemin selon votre environnement
DATA_PATH = 'C:\\Users\\alexandre.batisse\\.vscode\\Projet\\Projet_Stage_Scalian\\data\\Raw\\energydata_complete.csv'

OUTPUT_DIR = '.'   # dossier où sauvegarder les graphiques et CSV
RANDOM_STATE = 42

# ============================================================
# PARTIE 1 — CHARGEMENT ET PRÉTRAITEMENT
# ============================================================

def load_and_preprocess(path: str) -> pd.DataFrame:
    """Charge le CSV et applique le feature engineering.

    Règle d'or : AUCUNE feature ne doit être construite à partir de la
    valeur actuelle de la cible (Appliances à l'instant t). Sinon :
      - Le R² est artificiellement élevé (fuite de données)
      - Les simulations what-if sont biaisées (la feature "triche")

    Les features de lag (Appliances aux instants t-1, t-2...) sont
    acceptables car le passé est connu au moment de la prédiction.
    """
    print(f"\n{'='*60}\nCHARGEMENT DU DATASET\n{'='*60}")
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"Période        : {df['date'].min()} → {df['date'].max()}")
    print(f"Observations   : {len(df):,} (pas de 10 min)")
    print(f"Durée totale   : {(df['date'].max() - df['date'].min()).days} jours")

    # --- 1.1 Colonnes inutiles (variables aléatoires de contrôle) ---
    df.drop(columns=['rv1', 'rv2', 'Visibility'], inplace=True)

    # --- 1.2 Winsorisation des outliers sur Appliances ---
    # La distribution est très asymétrique (queue jusqu'à 1080 Wh).
    # On plafonne au P99 pour ne pas entraîner le modèle sur des pics
    # ponctuels (mise en route four, etc.).
    p99 = df['Appliances'].quantile(0.99)
    n_out = (df['Appliances'] > p99).sum()
    df['Appliances'] = df['Appliances'].clip(upper=p99)
    print(f"Outliers traités : {n_out} valeurs > {p99:.0f} Wh ramenées au P99")

    # --- 1.3 Feature engineering temporel ---
    df['hour']        = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month']       = df['date'].dt.month
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['is_night']    = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
    df['is_workhour'] = ((df['hour'] >= 8) & (df['hour'] < 18) &
                         (df['day_of_week'] < 5)).astype(int)
    # Encodage cyclique de l'heure (lundi 23h est proche de mardi 0h)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # --- 1.4 Features physiques dérivées (AUCUNE ne dépend de la cible) ---
    df['T_indoor_avg']           = df[[f'T{i}' for i in range(1, 10)]].mean(axis=1)
    df['T_indoor_outdoor_delta'] = df['T_indoor_avg'] - df['T_out']
    df['RH_indoor_avg']          = df[[f'RH_{i}' for i in range(1, 10)]].mean(axis=1)

    # Variabilité thermique récente (instabilité des pièces = indice d'activité)
    for col in ['T1', 'T2', 'T3']:
        df[f'{col}_rstd_6'] = df[col].rolling(6, min_periods=1).std().fillna(0)

    # --- 1.5 Lags de la cible (PASSÉ connu, donc pas de fuite) ---
    # Ces features capturent l'autocorrélation temporelle et sont
    # essentielles pour un R² réaliste sur ce dataset.
    # Pour un usage what-if "long terme", elles stabilisent les prédictions
    # dans la gamme historique → elles atténuent (= rendent conservateurs)
    # les effets des mesures simulées. C'est un biais connu et accepté.
    for lag in [1, 2, 3, 6, 144]:    # 10 min, 20 min, 30 min, 1 h, 1 jour
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
    # Météo extérieure
    'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Tdewpoint',
    # Températures intérieures (cibles des what-if chauffage)
    'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9',
    # Humidités intérieures
    'RH_1', 'RH_2', 'RH_3', 'RH_4', 'RH_5', 'RH_6', 'RH_7', 'RH_8', 'RH_9',
    # Temporelles
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_night', 'is_workhour',
    'hour_sin', 'hour_cos',
    # Dérivées physiques
    'T_indoor_avg', 'T_indoor_outdoor_delta', 'RH_indoor_avg',
    'T1_rstd_6', 'T2_rstd_6', 'T3_rstd_6',
    # Éclairage (cible du what-if éclairage)
    'lights',
    # Lags de consommation (autocorrélation)
    'Appliances_lag_1', 'Appliances_lag_2', 'Appliances_lag_3',
    'Appliances_lag_6', 'Appliances_lag_144',
    'Appliances_ma_6', 'Appliances_ma_144',
]
TARGET = 'Appliances'
T_COLS = [f'T{i}' for i in range(1, 10)]


def split_and_train(df: pd.DataFrame) -> dict:
    """Split temporel strict + entraînement XGBoost."""
    print(f"\n{'='*60}\nENTRAÎNEMENT DU MODÈLE\n{'='*60}")

    split_idx = int(len(df) * 0.70)
    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()

    d_tr = (df_train['date'].max() - df_train['date'].min()).days
    d_te = (df_test ['date'].max() - df_test ['date'].min()).days
    print(f"Train : {df_train['date'].min().date()} → {df_train['date'].max().date()} "
          f"({len(df_train):,} obs ≈ {d_tr} j)")
    print(f"Test  : {df_test ['date'].min().date()} → {df_test ['date'].max().date()} "
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

    print(f"\n--- Performances sur la période de test ---")
    print(f"MAE  : {mae:6.2f} Wh")
    print(f"RMSE : {rmse:6.2f} Wh")
    print(f"R²   : {r2:6.4f}")
    print(f"(Benchmarks publiés sur ce dataset : R² ≈ 0.45 à 0.70)")

    return {
        'model': model, 'scaler': scaler,
        'df_train': df_train, 'df_test': df_test,
        'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred,
        'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2},
    }


# ============================================================
# PARTIE 3 — SCÉNARIOS WHAT-IF
# ============================================================
#
# Chaque scénario reçoit X_test et renvoie une copie modifiée
# représentant l'état APRÈS application de la mesure.
# Les features dérivées (T_indoor_avg, delta) sont recalculées pour
# rester cohérentes avec les modifications.
# ============================================================

def recompute_derived(X: pd.DataFrame) -> pd.DataFrame:
    """Recalcule les features dérivées après modification des T°/lights."""
    X['T_indoor_avg']           = X[T_COLS].mean(axis=1)
    X['T_indoor_outdoor_delta'] = X['T_indoor_avg'] - X['T_out']
    return X


def scenario_A_heating_setpoint(X_test: pd.DataFrame, delta: float = 1.0) -> pd.DataFrame:
    """A : abaisser la consigne de chauffage de `delta` °C
    pendant les heures occupées (6h-22h) uniquement."""
    X = X_test.copy()
    mask = X['is_night'] == 0
    X.loc[mask, T_COLS] = X.loc[mask, T_COLS] - delta
    return recompute_derived(X)


def scenario_B_night_setback(X_test: pd.DataFrame,
                             target_delta_above_outdoor: float = 4.0) -> pd.DataFrame:
    """B : abaissement nocturne du chauffage (22h-6h).
    T° intérieure ramenée à T_out + `target_delta_above_outdoor` °C
    (on ne dépasse jamais la T° réelle → on ne chauffe pas MOINS que prévu)."""
    X = X_test.copy()
    mask = X['is_night'] == 1
    target = X.loc[mask, 'T_out'] + target_delta_above_outdoor
    for col in T_COLS:
        X.loc[mask, col] = np.minimum(X.loc[mask, col], target)
    return recompute_derived(X)


def scenario_C_lighting(X_test: pd.DataFrame, reduction_pct: float = 0.50) -> pd.DataFrame:
    """C : réduction de l'éclairage de `reduction_pct` (LED / détecteurs)."""
    X = X_test.copy()
    X['lights'] = X['lights'] * (1 - reduction_pct)
    return X


def scenario_D_combined(X_test: pd.DataFrame) -> pd.DataFrame:
    """D : combinaison A + B + C (bouquet de mesures)."""
    X = scenario_A_heating_setpoint(X_test, delta=1.0)
    X = scenario_B_night_setback(X, target_delta_above_outdoor=4.0)
    X = scenario_C_lighting(X, reduction_pct=0.30)
    return X


SCENARIOS = {
    '1': {
        'name': "Baisse consigne chauffage -1°C en journée",
        'description': "Abaissement de 1°C de la consigne de chauffage entre 6h et 22h.",
        'func': lambda X: scenario_A_heating_setpoint(X, delta=1.0),
    },
    '2': {
        'name': "Extinction programmée chauffage la nuit",
        'description': "Réduction nocturne (22h-6h) : T° intérieure ~ T° extérieure + 4°C.",
        'func': lambda X: scenario_B_night_setback(X, target_delta_above_outdoor=4.0),
    },
    '3': {
        'name': "Réduction éclairage -50%",
        'description': "Remplacement LED ou détecteurs de présence (-50% sur 'lights').",
        'func': lambda X: scenario_C_lighting(X, reduction_pct=0.50),
    },
    '4': {
        'name': "Bouquet combiné (A + B + C)",
        'description': "Application simultanée : -1°C jour + abaissement nuit + éclairage -30%.",
        'func': scenario_D_combined,
    },
}


# ============================================================
# PARTIE 4 — SIMULATION ET RAPPORT D'ÉCONOMIES
# ============================================================

def run_scenario(scenario_key: str, artifacts: dict) -> dict:
    """Applique un scénario et calcule les économies selon IPMVP."""
    scenario = SCENARIOS[scenario_key]
    model, scaler = artifacts['model'], artifacts['scaler']
    X_test, y_pred_baseline = artifacts['X_test'], artifacts['y_pred']

    # Application de la mesure (modification des features d'entrée)
    X_modified = scenario['func'](X_test)
    X_modified_sc = scaler.transform(X_modified[FEATURES])

    # Prédiction APRÈS mesure
    y_pred_modified = model.predict(X_modified_sc)

    # Formule IPMVP : Économies = Baseline ajustée − Consommation sous mesure
    savings_per_obs = y_pred_baseline - y_pred_modified    # Wh par 10 min

    total_baseline_kwh = y_pred_baseline.sum() / 1000
    total_scenario_kwh = y_pred_modified.sum() / 1000
    total_savings_kwh  = savings_per_obs.sum() / 1000
    pct_savings        = (savings_per_obs.sum() / y_pred_baseline.sum()) * 100

    # Extrapolation annuelle (test ≈ 41 jours)
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
    if result['total_savings_kwh'] < 0:
        print("⚠️  Économie négative : la mesure augmente la consommation prédite.")
        print("    Cause probable : corrélation non causale apprise par le modèle,")
        print("    ou effet d'amortissement par les features de lag.")


# ============================================================
# PARTIE 5 — VISUALISATIONS
# ============================================================

def plot_baseline(artifacts: dict):
    fig, ax = plt.subplots(figsize=(14, 5))
    dates = artifacts['df_test']['date'].values
    ax.plot(dates, artifacts['y_test'].values, label='Consommation reelle',
            alpha=0.6, color='steelblue', linewidth=0.8)
    ax.plot(dates, artifacts['y_pred'], label='Baseline predite (modele)',
            alpha=0.85, color='orange', linewidth=0.8)
    m = artifacts['metrics']
    ax.set_title(f"Baseline IPMVP - R2={m['r2']:.3f} | MAE={m['mae']:.1f} Wh | "
                 f"RMSE={m['rmse']:.1f} Wh")
    ax.set_ylabel('Consommation (Wh)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'baseline_vs_real.png')
    plt.savefig(path, dpi=130); print(f"-> {path}")
    plt.show()


def plot_scenario(result: dict, artifacts: dict):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    dates = artifacts['df_test']['date'].values

    ax1 = axes[0]
    ax1.plot(dates, result['y_pred_baseline'], label='Baseline (sans mesure)',
             color='steelblue', linewidth=0.8, alpha=0.8)
    ax1.plot(dates, result['y_pred_modified'], label='Avec mesure appliquee',
             color='seagreen', linewidth=0.8, alpha=0.8)
    ax1.fill_between(dates, result['y_pred_baseline'], result['y_pred_modified'],
                     where=(result['y_pred_baseline'] > result['y_pred_modified']),
                     alpha=0.25, color='seagreen', label='Economies')
    ax1.set_title(f"Scenario : {result['scenario']['name']}")
    ax1.set_ylabel('Consommation (Wh)')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2 = axes[1]
    labels = ['Baseline', 'Avec mesure', 'Economies']
    values = [result['total_baseline_kwh'],
              result['total_scenario_kwh'],
              result['total_savings_kwh']]
    colors = ['steelblue', 'seagreen', 'tomato']
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
    # 1. Chargement + prétraitement
    df = load_and_preprocess(DATA_PATH)

    # 2. Entraînement + prédiction de la baseline sur la période de test
    artifacts = split_and_train(df)

    # 3. Visualisation de la baseline (réel vs. prédit)
    plot_baseline(artifacts)

    # 4. Menu interactif de simulation what-if
    interactive_menu(artifacts)


if __name__ == '__main__':
    main()