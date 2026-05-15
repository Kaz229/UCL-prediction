"""
features.py — Feature engineering à partir des données brutes UCL.

Lit les données depuis DuckDB (tables raw_*), applique les transformations
et écrit le résultat dans la table 'features' de DuckDB.

Transformations appliquées :
  1. Variable cible      : encode le résultat (2=victoire dom., 1=nul, 0=défaite dom.)
  2. Forme rolling       : stats sur les N derniers matchs UCL (fenêtre adaptative par phase)
  3. Forme contextuelle  : forme à domicile / à l'extérieur séparées
  4. Standings UCL       : classement phase de ligue (knockout uniquement, pas de fuite)
  5. Stats domestiques   : classement championnat national (top 5 ligues)
  6. Features diff.      : écarts home - away pour chaque groupe de features
  7. Encodage de phase   : ordinal (1=ligue → 6=finale)

Point clé anti-fuite : la forme rolling est calculée sur les matchs
PRÉCÉDANT le match courant. L'historique est mis à jour après chaque itération.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.loader import get_connection
from config.settings import (
    PHASE_ORDER, KNOCKOUT_PHASES, PHASE_WINDOW, DB_PATH
)


# ──────────────────────────────────────────────
# Chargement depuis DuckDB
# ──────────────────────────────────────────────

def load_matches(conn=None) -> pd.DataFrame:
    """Charge les matchs UCL depuis DuckDB, triés par date."""
    if conn is None:
        conn = get_connection()
    return conn.execute(
        "SELECT * FROM raw_ucl_matches ORDER BY date"
    ).df()


def load_standings(conn=None) -> pd.DataFrame:
    """Charge le classement UCL phase de ligue depuis DuckDB."""
    if conn is None:
        conn = get_connection()
    try:
        return conn.execute("SELECT * FROM raw_ucl_standings").df()
    except Exception:
        return pd.DataFrame()


def load_domestic_standings(conn=None) -> pd.DataFrame:
    """Charge les classements domestiques depuis DuckDB."""
    if conn is None:
        conn = get_connection()
    try:
        return conn.execute("SELECT * FROM raw_domestic_standings").df()
    except Exception:
        return pd.DataFrame()


# ──────────────────────────────────────────────
# 1. Variable cible
# ──────────────────────────────────────────────

def add_result(df: pd.DataFrame) -> pd.DataFrame:
    """Encode le résultat : 2=victoire dom., 1=nul, 0=défaite dom."""
    def _label(row):
        if row["home_goals"] > row["away_goals"]:
            return 2
        elif row["home_goals"] == row["away_goals"]:
            return 1
        return 0
    df["result"] = df.apply(_label, axis=1)
    return df


# ──────────────────────────────────────────────
# 2 & 3. Forme rolling (globale + contextuelle)
# ──────────────────────────────────────────────

def _rolling_stats(history: list, n: int) -> dict:
    """Stats sur les n derniers matchs d'un historique."""
    last_n = history[-n:] if len(history) >= n else history
    return {
        "gf":     sum(x[1] for x in last_n),
        "ga":     sum(x[2] for x in last_n),
        "pts":    sum(x[3] for x in last_n),
        "played": len(last_n),
    }


def add_form_features(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Forme récente : stats calculées sur les matchs PRÉCÉDANT chaque match.
    - Forme globale  : tous matchs UCL confondus
    - Forme dom/ext  : séparée selon le contexte (domicile ou extérieur)
    La fenêtre s'adapte à la phase via PHASE_WINDOW.
    """
    history      = {}
    history_home = {}
    history_away = {}
    rows = []

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hg, ag     = row["home_goals"], row["away_goals"]
        window     = PHASE_WINDOW.get(row["phase"], n)

        home_stats       = _rolling_stats(history.get(home, []), window)
        away_stats       = _rolling_stats(history.get(away, []), window)
        home_home_stats  = _rolling_stats(history_home.get(home, []), window)
        away_away_stats  = _rolling_stats(history_away.get(away, []), window)

        rows.append({
            **row.to_dict(),
            "home_form_gf":        home_stats["gf"],
            "home_form_ga":        home_stats["ga"],
            "home_form_pts":       home_stats["pts"],
            "away_form_gf":        away_stats["gf"],
            "away_form_ga":        away_stats["ga"],
            "away_form_pts":       away_stats["pts"],
            "home_form_at_home_gf":  home_home_stats["gf"],
            "home_form_at_home_ga":  home_home_stats["ga"],
            "home_form_at_home_pts": home_home_stats["pts"],
            "away_form_at_away_gf":  away_away_stats["gf"],
            "away_form_at_away_ga":  away_away_stats["ga"],
            "away_form_at_away_pts": away_away_stats["pts"],
        })

        if hg > ag:
            hp, ap = 3, 0
        elif hg == ag:
            hp, ap = 1, 1
        else:
            hp, ap = 0, 3

        history.setdefault(home, []).append((row["date"], hg, ag, hp))
        history.setdefault(away, []).append((row["date"], ag, hg, ap))
        history_home.setdefault(home, []).append((row["date"], hg, ag, hp))
        history_away.setdefault(away, []).append((row["date"], ag, hg, ap))

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 4. Standings UCL
# ──────────────────────────────────────────────

def add_standings_features(df: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
    """
    Classement UCL phase de ligue pour chaque équipe.
    Appliqué uniquement aux phases knockout pour éviter toute fuite de données
    (le classement final n'est pas connu pendant la phase de ligue).
    """
    standing_cols = ["position", "points", "goal_diff", "goals_for", "goals_against"]

    for col in standing_cols:
        df[f"home_standing_{col}"] = 0
        df[f"away_standing_{col}"] = 0

    if standings.empty:
        return df

    idx = standings.set_index("team")
    mask = df["phase"].isin(KNOCKOUT_PHASES)

    for side in ("home", "away"):
        for col in standing_cols:
            df.loc[mask, f"{side}_standing_{col}"] = (
                df.loc[mask, f"{side}_team"].map(idx[col]).fillna(0)
            )

    df["diff_standing_pts"] = df["home_standing_points"]   - df["away_standing_points"]
    df["diff_standing_pos"] = df["home_standing_position"] - df["away_standing_position"]
    df["diff_standing_gd"]  = df["home_standing_goal_diff"] - df["away_standing_goal_diff"]
    return df


# ──────────────────────────────────────────────
# 5. Stats ligue domestique
# ──────────────────────────────────────────────

def add_domestic_features(df: pd.DataFrame, domestic: pd.DataFrame) -> pd.DataFrame:
    """
    Classement domestique pour chaque équipe UCL (jointure par nom d'équipe).
    Les équipes hors top 5 championnats auront des valeurs à 0.
    Pas de restriction de phase : les stats domestiques ne constituent pas
    une fuite de données pour les matchs UCL.
    """
    dom_stats = ["position", "points", "goal_diff"]

    for col in dom_stats:
        df[f"home_domestic_{col}"] = 0
        df[f"away_domestic_{col}"] = 0

    if domestic.empty:
        return df

    idx = domestic.set_index("team")

    for side in ("home", "away"):
        for col in dom_stats:
            df[f"{side}_domestic_{col}"] = (
                df[f"{side}_team"].map(idx[col]).fillna(0)
            )

    df["diff_domestic_pts"] = df["home_domestic_points"]   - df["away_domestic_points"]
    df["diff_domestic_pos"] = df["home_domestic_position"] - df["away_domestic_position"]
    df["diff_domestic_gd"]  = df["home_domestic_goal_diff"] - df["away_domestic_goal_diff"]
    return df


# ──────────────────────────────────────────────
# 6. Features différentielles (forme)
# ──────────────────────────────────────────────

def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Différences home - away pour les features de forme (globale et contextuelle)."""
    df["diff_form_pts"]     = df["home_form_pts"]         - df["away_form_pts"]
    df["diff_form_gf"]      = df["home_form_gf"]          - df["away_form_gf"]
    df["diff_form_ga"]      = df["home_form_ga"]          - df["away_form_ga"]
    df["diff_form_ctx_pts"] = df["home_form_at_home_pts"] - df["away_form_at_away_pts"]
    df["diff_form_ctx_gf"]  = df["home_form_at_home_gf"]  - df["away_form_at_away_gf"]
    df["diff_form_ctx_ga"]  = df["home_form_at_home_ga"]  - df["away_form_at_away_ga"]
    return df


# ──────────────────────────────────────────────
# 7. Encodage de la phase
# ──────────────────────────────────────────────

def add_phase_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Encode la phase en valeur ordinale (1=ligue → 6=finale)."""
    df["phase_encoded"] = df["phase"].map(PHASE_ORDER).fillna(0).astype(int)
    return df


# ──────────────────────────────────────────────
# Pipeline complet
# ──────────────────────────────────────────────

def build_features(conn=None) -> pd.DataFrame:
    """
    Charge les données depuis DuckDB, applique toutes les transformations
    et retourne le DataFrame prêt pour l'entraînement.
    """
    if conn is None:
        conn = get_connection()

    df       = load_matches(conn)
    standings = load_standings(conn)
    domestic  = load_domestic_standings(conn)

    df = add_result(df)
    df = add_form_features(df, n=5)
    df = add_standings_features(df, standings)
    df = add_domestic_features(df, domestic)
    df = add_diff_features(df)
    df = add_phase_encoding(df)
    return df


def save_to_db(df: pd.DataFrame, conn=None) -> None:
    """Écrit les features dans la table DuckDB 'features'."""
    if conn is None:
        conn = get_connection()
    conn.register("_features", df)
    conn.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM _features")
    conn.unregister("_features")
    print(f"  Table 'features' : {len(df)} lignes x {df.shape[1]} colonnes")


def run(conn=None) -> pd.DataFrame:
    """Lance le pipeline complet et persiste le résultat dans DuckDB."""
    if conn is None:
        conn = get_connection()

    print("[Features] Construction du dataset...")
    df = build_features(conn)
    save_to_db(df, conn)

    print(f"\n  {df.shape[0]} matchs x {df.shape[1]} colonnes")
    print("\n  Distribution des résultats :")
    print("  " + str(df["result"].value_counts()
                     .rename({2: "Victoire dom.", 1: "Nul", 0: "Défaite dom."})
                     .to_dict()))
    print("\n[Features] Terminé.")
    return df


if __name__ == "__main__":
    run()
