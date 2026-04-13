import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print("=" * 50)
print("  TEST ENVIRONNEMENT — Energy Analysis Project")
print("=" * 50)
print(f"  NumPy       : {np.__version__}")
print(f"  Pandas      : {pd.__version__}")
print(f"  Scikit-learn: {__import__('sklearn').__version__}")
print(f"  Matplotlib  : {__import__('matplotlib').__version__}")
print("=" * 50)

# 1. NumPy : génération de données de consommation simulées 
np.random.seed(42)
heures = np.arange(0, 168)  # 1 semaine en heures

# Profil de consommation journalier (en kWh)
profil_journalier = 10 + 8 * np.sin((heures % 24 - 8) * np.pi / 12)
bruit = np.random.normal(0, 1.2, size=168)
consommation = np.clip(profil_journalier + bruit, 0, None)

# Injection de 5 anomalies
anomalies_idx = [15, 42, 89, 110, 155]
consommation[anomalies_idx] += np.random.uniform(15, 25, size=5)

print("\n✅ NumPy OK — données simulées générées (168h, 5 anomalies injectées)")

# 2. Pandas : mise en DataFrame et analyse
dates = pd.date_range(start="2026-04-07", periods=168, freq="h")
df = pd.DataFrame({
    "timestamp": dates,
    "consommation_kwh": consommation,
    "jour_semaine": dates.day_name(),
    "heure": dates.hour
})

stats = df["consommation_kwh"].describe().round(2)
moyenne_par_heure = df.groupby("heure")["consommation_kwh"].mean()

print(f"\n✅ Pandas OK — DataFrame créé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"   Consommation moyenne : {stats['mean']} kWh | Max : {stats['max']} kWh")

# 3. Scikit-learn : détection d'anomalies avec Isolation Forest
scaler = StandardScaler()
X = scaler.fit_transform(df[["consommation_kwh", "heure"]])

iso_forest = IsolationForest(contamination=0.04, random_state=42)
df["anomalie"] = iso_forest.fit_predict(X)
df["anomalie_label"] = df["anomalie"].map({1: "Normal", -1: "⚠️ Anomalie"})

n_anomalies = (df["anomalie"] == -1).sum()
print(f"\n✅ Scikit-learn OK — IsolationForest exécuté")
print(f"   Anomalies détectées : {n_anomalies} sur {len(df)} points")

# 4. Matplotlib : visualisation
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("Test Environnement — Consommation Énergétique (simulée)", fontsize=14, fontweight="bold")

# Graphe 1 : série temporelle + anomalies
ax1 = axes[0]
normaux = df[df["anomalie"] == 1]
detectes = df[df["anomalie"] == -1]
ax1.plot(df["timestamp"], df["consommation_kwh"], color="#4A90D9", linewidth=0.9, label="Consommation (kWh)")
ax1.scatter(detectes["timestamp"], detectes["consommation_kwh"], color="crimson", zorder=5, s=60, label=f"Anomalies détectées ({n_anomalies})")
ax1.set_ylabel("Consommation (kWh)")
ax1.set_title("Série temporelle sur 1 semaine")
ax1.legend()
ax1.grid(alpha=0.3)

# Graphe 2 : profil moyen par heure
ax2 = axes[1]
ax2.bar(moyenne_par_heure.index, moyenne_par_heure.values, color="#2ECC71", edgecolor="white", linewidth=0.5)
ax2.set_xlabel("Heure de la journée")
ax2.set_ylabel("Consommation moyenne (kWh)")
ax2.set_title("Profil moyen de consommation par heure")
ax2.set_xticks(range(0, 24))
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("test_environnement.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✅ Matplotlib OK — graphique généré et sauvegardé (test_environnement.png)")
print("\n" + "=" * 50)
print("  ✅ ENVIRONNEMENT 100% OPÉRATIONNEL")
print("=" * 50)