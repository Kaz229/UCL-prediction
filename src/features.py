"""
features.py — Feature engineering à partir des données brutes UCL
"""

import pandas as pd
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

PHASE_ORDER = {
    "LEAGUE_STAGE": 1,
    "PLAYOFFS": 2,
    "LAST_16": 3,
    "QUARTER_FINALS": 4,
    "SEMI_FINALS": 5,
    "FINAL": 6,
}


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_PATH, "ucl_matches.csv"), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# 1. Variable cible
# ──────────────────────────────────────────────

def add_result(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode le résultat du match :
    2 = victoire domicile, 1 = nul, 0 = défaite domicile
    """
    def _label(row):
        if row["home_goals"] > row["away_goals"]:
            return 2
        elif row["home_goals"] == row["away_goals"]:
            return 1
        return 0

    df["result"] = df.apply(_label, axis=1)
    return df


# ──────────────────────────────────────────────
# 2. Forme récente (rolling sur les 5 derniers matchs)
# ──────────────────────────────────────────────

def _team_history(df: pd.DataFrame) -> dict:
    """
    Construit un historique ordonné par équipe :
    {team: [(date, goals_for, goals_against, points), ...]}
    """
    history = {}

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hg, ag = row["home_goals"], row["away_goals"]

        if hg > ag:
            hp, ap = 3, 0
        elif hg == ag:
            hp, ap = 1, 1
        else:
            hp, ap = 0, 3

        for team, gf, ga, pts in [(home, hg, ag, hp), (away, ag, hg, ap)]:
            if team not in history:
                history[team] = []
            history[team].append((row["date"], gf, ga, pts))

    return history


def _rolling_stats(history: list, n: int = 5) -> dict:
    """Calcule les stats sur les n derniers matchs d'un historique."""
    last_n = history[-n:] if len(history) >= n else history
    goals_for = sum(x[1] for x in last_n)
    goals_against = sum(x[2] for x in last_n)
    points = sum(x[3] for x in last_n)
    return {
        "gf": goals_for,
        "ga": goals_against,
        "pts": points,
        "played": len(last_n),
    }


def add_form_features(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Ajoute les features de forme récente pour chaque équipe avant chaque match.
    Les stats sont calculées sur les matchs PRÉCÉDANT le match courant.
    """
    history = {}
    rows = []

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hg, ag = row["home_goals"], row["away_goals"]

        # Stats avant ce match
        home_stats = _rolling_stats(history.get(home, []), n)
        away_stats = _rolling_stats(history.get(away, []), n)

        rows.append({
            **row.to_dict(),
            "home_form_gf": home_stats["gf"],
            "home_form_ga": home_stats["ga"],
            "home_form_pts": home_stats["pts"],
            "away_form_gf": away_stats["gf"],
            "away_form_ga": away_stats["ga"],
            "away_form_pts": away_stats["pts"],
        })

        # Mise à jour de l'historique après le match
        if hg > ag:
            hp, ap = 3, 0
        elif hg == ag:
            hp, ap = 1, 1
        else:
            hp, ap = 0, 3

        for team, gf, ga, pts in [(home, hg, ag, hp), (away, ag, hg, ap)]:
            if team not in history:
                history[team] = []
            history[team].append((row["date"], gf, ga, pts))

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 3. Features différentielles
# ──────────────────────────────────────────────

def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les différences home - away pour les features de forme."""
    df["diff_form_pts"] = df["home_form_pts"] - df["away_form_pts"]
    df["diff_form_gf"] = df["home_form_gf"] - df["away_form_gf"]
    df["diff_form_ga"] = df["home_form_ga"] - df["away_form_ga"]
    return df


# ──────────────────────────────────────────────
# 4. Encodage de la phase
# ──────────────────────────────────────────────

def add_phase_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Encode la phase du tournoi en valeur ordinale."""
    df["phase_encoded"] = df["phase"].map(PHASE_ORDER).fillna(0).astype(int)
    return df


# ──────────────────────────────────────────────
# 5. Pipeline complet
# ──────────────────────────────────────────────

def build_features() -> pd.DataFrame:
    """
    Pipeline complet : charge les données brutes et retourne
    un DataFrame prêt pour l'entraînement du modèle.
    """
    df = load_matches()
    df = add_result(df)
    df = add_form_features(df, n=5)
    df = add_diff_features(df)
    df = add_phase_encoding(df)
    return df


def save_processed(df: pd.DataFrame, filename: str) -> None:
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    path = os.path.join(PROCESSED_PATH, filename)
    df.to_csv(path, index=False)
    print(f"Sauvegardé : {path} ({len(df)} lignes)")


if __name__ == "__main__":
    df = build_features()

    print(f"\nDataset : {df.shape[0]} matchs x {df.shape[1]} colonnes")
    print("\nAperçu des features :")
    print(df[["date", "home_team", "away_team", "home_form_pts", "away_form_pts",
              "diff_form_pts", "phase_encoded", "result"]].head(10))

    print("\nDistribution des résultats :")
    print(df["result"].value_counts().rename({2: "Victoire dom.", 1: "Nul", 0: "Défaite dom."}))

    save_processed(df, "ucl_features.csv")
