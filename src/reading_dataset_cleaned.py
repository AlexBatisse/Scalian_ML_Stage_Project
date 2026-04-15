"""
╔══════════════════════════════════════════════════════════════╗
║         Energy Data Viewer — energydata_cleaned.csv          ║
╚══════════════════════════════════════════════════════════════╝
Affichage complet et lisible des données nettoyées en console.
Navigation par pages, stats résumées, colonnes organisées.
"""

import pandas as pd
import os
import sys

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CSV_PATH  = r"C:\Users\alexandre.batisse\.vscode\Projet\Projet_Stage_Scalian\data\Processed\energydata_cleaned.csv"
PAGE_SIZE = 20      # lignes par page
FLOAT_DEC = 2       # décimales pour les floats

# Couleurs ANSI
C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_CYAN   = "\033[96m"
C_GREEN  = "\033[92m"
C_YELLOW = "\033[93m"
C_RED    = "\033[91m"
C_BLUE   = "\033[94m"
C_GREY   = "\033[90m"
C_WHITE  = "\033[97m"


def clear():
    os.system("cls" if os.name == "nt" else "clear")

def banner():
    print(f"""
{C_CYAN}{C_BOLD}╔══════════════════════════════════════════════════════════════════════╗
║           ⚡  Energy Data Viewer — energydata_cleaned.csv   ⚡        ║
╚══════════════════════════════════════════════════════════════════════╝{C_RESET}
""")

# ─────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────
def load_data():
    try:
        # index_col=0 : la 1ère colonne (sans nom) est l\'index — c\'est le timestamp sauvegardé par dataset_cleaning.py
        df = pd.read_csv(CSV_PATH, index_col=0)
        df.index.name = "timestamp"
        df = df.reset_index()                                      # remet timestamp comme colonne normale
        df["timestamp"] = pd.to_datetime(df["timestamp"])         # parse en datetime
        df.columns = [c.strip() for c in df.columns]
        float_cols = df.select_dtypes(include="float").columns
        df[float_cols] = df[float_cols].round(FLOAT_DEC)
        return df
    except FileNotFoundError:
        print(f"{C_RED}Erreur : fichier \'{CSV_PATH}\' introuvable.{C_RESET}")
        print(f"{C_GREY}Placez le fichier dans le même dossier que ce script.{C_RESET}")
        sys.exit(1)

# ─────────────────────────────────────────
# AFFICHAGE RÉSUMÉ
# ─────────────────────────────────────────
def print_summary(df):
    banner()
    n_rows, n_cols = df.shape
    ts_min = df["timestamp"].min().strftime("%Y-%m-%d %H:%M")
    ts_max = df["timestamp"].max().strftime("%Y-%m-%d %H:%M")
    duration = df["timestamp"].max() - df["timestamp"].min()

    print(f"  {C_BOLD}📁 Fichier      :{C_RESET} {C_WHITE}{CSV_PATH}{C_RESET}")
    print(f"  {C_BOLD}📊 Lignes       :{C_RESET} {C_GREEN}{n_rows:,}{C_RESET}")
    print(f"  {C_BOLD}📐 Colonnes     :{C_RESET} {C_GREEN}{n_cols}{C_RESET}")
    print(f"  {C_BOLD}📅 Période      :{C_RESET} {C_CYAN}{ts_min}{C_RESET}  →  {C_CYAN}{ts_max}{C_RESET}")
    print(f"  {C_BOLD}⏱  Durée        :{C_RESET} {C_YELLOW}{duration.days} jours{C_RESET}")
    print(f"  {C_BOLD}🔁 Fréquence    :{C_RESET} {C_YELLOW}10 min{C_RESET}")
    print(f"  {C_BOLD}✅ Statut       :{C_RESET} {C_GREEN}Données nettoyées (rv1/rv2 supprimés, outliers écrêtés){C_RESET}")
    print(f"\n  {C_BOLD}📋 Colonnes disponibles :{C_RESET}")

    # Groupes logiques — rv1/rv2 absents du fichier cleaned
    groups = {
        "🕐 Timestamp"         : ["timestamp"],
        "🔌 Conso. électrique" : ["Appliances", "lights"],
        "🌡  Températures int.": [c for c in df.columns if c.startswith("T") and c[1:].isdigit()],
        "💧 Humidité int.     ": [c for c in df.columns if c.startswith("RH_") and c[3:].isdigit()],
        "🌤  Météo extérieure" : ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"],
    }
    for group, cols in groups.items():
        cols_in_df = [c for c in cols if c in df.columns]
        if cols_in_df:
            print(f"\n   {C_BOLD}{group}{C_RESET}")
            for c in cols_in_df:
                dtype = str(df[c].dtype)
                print(f"      {C_GREY}•{C_RESET} {C_WHITE}{c:<18}{C_RESET} {C_GREY}({dtype}){C_RESET}")

    print(f"\\n{C_GREY}{'─'*72}{C_RESET}")
    print(f"  {C_BOLD}Statistiques rapides — consommation Appliances (Wh) :{C_RESET}")
    a = df["Appliances"]
    print(f"    Min={C_GREEN}{a.min()}{C_RESET}  Max={C_RED}{a.max()}{C_RESET}  "
          f"Moy={C_YELLOW}{a.mean():.1f}{C_RESET}  Méd={C_YELLOW}{a.median():.1f}{C_RESET}  "
          f"Std={C_GREY}{a.std():.1f}{C_RESET}")

    vals = a.value_counts().nlargest(5)
    print(f"\n {C_BOLD}Top 5 valeurs les plus fréquentes (Appliances) :{C_RESET}")
    for val, cnt in vals.items():
        bar = "█" * min(int(cnt / 100), 30)
        print(f"    {C_CYAN}{val:>6} Wh{C_RESET}  {C_GREY}{bar}{C_RESET}  {cnt} fois")

    # Stats NaN résiduels
    nan_total = df.isnull().sum().sum()
    nan_color = C_GREEN if nan_total == 0 else C_YELLOW
    print(f"\n {C_BOLD}🔍 Valeurs manquantes (NaN) :{C_RESET} {nan_color}{nan_total}{C_RESET}")

    print(f"\\n{C_GREY}{'─'*72}{C_RESET}\\n")

# ─────────────────────────────────────────
# AFFICHAGE PAGE DE DONNÉES
# ─────────────────────────────────────────
# Colonnes affichées dans la vue paginée (sous-ensemble lisible)
DISPLAY_COLS = ["timestamp", "Appliances", "lights",
                "T1", "RH_1", "T2", "RH_2", "T6", "RH_6",
                "T_out", "RH_out", "Windspeed", "Visibility", "Tdewpoint",
                "Press_mm_hg"]

COL_WIDTHS = {
    "timestamp"  : 19,
    "Appliances" : 10,
    "lights"     : 7,
    "T1"         : 7,
    "RH_1"       : 7,
    "T2"         : 7,
    "RH_2"       : 7,
    "T6"         : 7,
    "RH_6"       : 7,
    "T_out"      : 7,
    "RH_out"     : 7,
    "Windspeed"  : 10,
    "Visibility" : 10,
    "Tdewpoint"  : 10,
    "Press_mm_hg": 12,
}

COL_LABELS = {
    "timestamp"  : "Timestamp",
    "Appliances" : "App.(Wh)",
    "lights"     : "Lum.",
    "T1"         : "T1(°C)",
    "RH_1"       : "RH1(%)",
    "T2"         : "T2(°C)",
    "RH_2"       : "RH2(%)",
    "T6"         : "T6(°C)",
    "RH_6"       : "RH6(%)",
    "T_out"      : "T_ext",
    "RH_out"     : "RH_ext",
    "Windspeed"  : "Vent(m/s)",
    "Visibility" : "Visib.",
    "Tdewpoint"  : "T_rosée",
    "Press_mm_hg": "Pression",
}

def make_header():
    parts = []
    for col in DISPLAY_COLS:
        w = COL_WIDTHS.get(col, 10)
        lbl = COL_LABELS.get(col, col)
        parts.append(f"{C_BOLD}{C_BLUE}{lbl:^{w}}{C_RESET}")
    sep = f"{C_GREY} │ {C_RESET}"
    return sep.join(parts)

def make_separator():
    parts = [f"{C_GREY}{'-' * COL_WIDTHS.get(c, 10)}{C_RESET}" for c in DISPLAY_COLS]
    return f"{C_GREY}─┼─{C_RESET}".join(parts)

def format_row(row, i):
    """Formate une ligne avec couleur alternée."""
    color = C_WHITE if i % 2 == 0 else C_GREY
    parts = []
    for col in DISPLAY_COLS:
        w = COL_WIDTHS.get(col, 10)
        val = row[col]
        if col == "timestamp":
            s = str(val)[:19]
            parts.append(f"{C_CYAN}{s:<{w}}{C_RESET}")
        elif col == "Appliances":
            v = int(val)
            if v > 400:
                c = C_RED
            elif v > 150:
                c = C_YELLOW
            else:
                c = C_GREEN
            parts.append(f"{c}{str(v):>{w}}{C_RESET}")
        elif col == "lights":
            v = int(val)
            c = C_YELLOW if v > 0 else C_GREY
            parts.append(f"{c}{str(v):>{w}}{C_RESET}")
        else:
            parts.append(f"{color}{str(val):>{w}}{C_RESET}")
    sep = f"{C_GREY} │ {C_RESET}"
    return sep.join(parts)

def print_page(df, page, total_pages, filter_col=None, filter_val=None):
    clear()
    banner()

    display_df = df
    label_extra = ""
    if filter_col and filter_val is not None:
        display_df = df[df[filter_col] == filter_val]
        label_extra = f"  {C_YELLOW}Filtre : {filter_col} = {filter_val}  ({len(display_df)} lignes){C_RESET}"

    total_pages_cur = max(1, (len(display_df) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(page, total_pages_cur - 1)

    start = page * PAGE_SIZE
    end   = start + PAGE_SIZE
    chunk = display_df.iloc[start:end]

    print(f"  {C_BOLD}Page {C_CYAN}{page+1}{C_RESET}{C_BOLD}/{C_CYAN}{total_pages_cur}{C_RESET}"
          f"   {C_GREY}Lignes {start+1}–{min(end, len(display_df))} / {len(display_df):,}{C_RESET}"
          + (f"   {label_extra}" if label_extra else ""))
    print()

    print("  " + make_header())
    print("  " + make_separator())
    for i, (_, row) in enumerate(chunk.iterrows()):
        print("  " + format_row(row, i))
    print("  " + make_separator())
    print()

    print(f"  {C_BOLD}Légende Appliances :{C_RESET}  "
          f"{C_GREEN}■ ≤150 Wh{C_RESET}  {C_YELLOW}■ 151–400 Wh{C_RESET}  {C_RED}■ >400 Wh{C_RESET}")
    print()

    print(f"  {C_GREY}{'─'*60}{C_RESET}")
    print(f"  {C_BOLD}Navigation :{C_RESET}  "
          f"{C_CYAN}[n]{C_RESET}=suivant  "
          f"{C_CYAN}[p]{C_RESET}=précédent  "
          f"{C_CYAN}[g <num>]{C_RESET}=aller page  "
          f"{C_CYAN}[d <date>]{C_RESET}=chercher date")
    print(f"               "
          f"{C_CYAN}[s]{C_RESET}=résumé stats  "
          f"{C_CYAN}[a]{C_RESET}=toutes colonnes  "
          f"{C_CYAN}[q]{C_RESET}=quitter")
    print(f"  {C_GREY}{'─'*60}{C_RESET}")


    return page, total_pages_cur, display_df

def print_all_columns(df, page):
    """Affiche toutes les colonnes pour les lignes de la page courante."""
    clear()
    banner()
    start = page * PAGE_SIZE
    end   = start + PAGE_SIZE
    chunk = df.iloc[start:end]

    print(f"  {C_BOLD}{C_CYAN}Toutes les colonnes — lignes {start+1} à {min(end, len(df))}{C_RESET}\\n")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")

    print(chunk.to_string(index=False))
    print(f"\n {C_GREY}[Appuyez sur Entrée pour revenir]{C_RESET}")
    input()

# ─────────────────────────────────────────
# BOUCLE PRINCIPALE
# ─────────────────────────────────────────
def main():
    df = load_data()
    total_pages = max(1, (len(df) + PAGE_SIZE - 1) // PAGE_SIZE)

    clear()
    print_summary(df)
    print(f"  {C_GREY}Appuyez sur Entrée pour parcourir les données...{C_RESET}")
    input()

    page = 0

    while True:
        page, total_pages, _ = print_page(df, page, total_pages)

        cmd = input(f"  {C_BOLD}> {C_RESET}").strip().lower()

        if cmd == "q":
            clear()
            print(f"\n {C_GREEN}Au revoir !{C_RESET}\\n")
            break

        elif cmd == "n":
            if page < total_pages - 1:
                page += 1

        elif cmd == "p":
            if page > 0:
                page -= 1

        elif cmd.startswith("g "):
            try:
                num = int(cmd.split()[1]) - 1
                if 0 <= num < total_pages:
                    page = num
                else:
                    print(f"  {C_RED}Page invalide (1–{total_pages}){C_RESET}")
                    input()
            except (ValueError, IndexError):
                pass

        elif cmd.startswith("d "):
            date_str = cmd[2:].strip()
            found = df[df["timestamp"].astype(str).str.startswith(date_str)]
            if not found.empty:
                page = found.index[0] // PAGE_SIZE
            else:
                print(f"  {C_RED}Aucune ligne trouvée pour \'{date_str}\'{C_RESET}")
                input()

        elif cmd == "s":
            clear()
            print_summary(df)
            input(f"  {C_GREY}[Entrée pour continuer]{C_RESET}")

        elif cmd == "a":
            print_all_columns(df, page)

        else:
            if cmd == "" and page < total_pages - 1:
                page += 1


if __name__ == "__main__":
    main()