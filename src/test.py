import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

DATA_PATH   = "C:\\Users\\alexandre.batisse\\.vscode\\Projet\\Projet_Stage_Scalian\\data\\Raw\\energydata_complete.csv"

print("\n═══ 1. CHARGEMENT ═══")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df_mai = df[df['date'].dt.month == 5]
somme_appliance = df['Appliances'].sum() / 1000
valeur_moy_temp_cuisine = df['T1'].mean()
moy_mai_T_kitchen = df_mai['T1'].mean()

for i in range(1, 6) :
    df_month = df[df['date'].dt.month == i]
    moy_per_month_T1 = df_month['T1'].mean()
    print(f"moyenne de température vaut : {moy_per_month_T1} , pour le mois n° : {i}")
