"""
scraper.py — Collecte des données UCL via football-data.org API
"""

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Ligues domestiques couvertes par l'API (tier gratuit)
DOMESTIC_LEAGUES = {
    "PL":  "Premier League",
    "PD":  "Primera Division",
    "BL1": "Bundesliga",
    "SA":  "Serie A",
    "FL1": "Ligue 1",
}

# Saisons historiques UCL à collecter (en plus de la saison courante)
HISTORICAL_SEASONS = [2019, 2020, 2021, 2022, 2023]

# Normalisation des noms de phase : l'UCL a changé de format en 2024
PHASE_NORMALIZE = {
    "GROUP_STAGE": "LEAGUE_STAGE",
    "ROUND_OF_16": "LAST_16",
}


def _get(endpoint: str, params: dict = None) -> dict:
    """GET sur l'API football-data.org avec gestion du rate limit."""
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 429:
        print("Rate limit atteint, attente 60s...")
        time.sleep(60)
        response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json()


# ──────────────────────────────────────────────
# 1. Matchs UCL (variable cible)
# ──────────────────────────────────────────────

def scrape_ucl_matches(season: int = 2024) -> pd.DataFrame:
    """
    Récupère tous les matchs UCL de la saison.
    season=2024 correspond à la saison 2024-2025.
    Retourne : date, phase, home_team, away_team, home_goals, away_goals, status
    """
    print(f"Scraping matchs UCL {season}...")
    data = _get("competitions/CL/matches", params={"season": season})

    rows = []
    for m in data.get("matches", []):
        rows.append({
            "match_id": m["id"],
            "date": m["utcDate"][:10],
            "phase": m["stage"],
            "matchday": m.get("matchday"),
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
            "home_goals": m["score"]["fullTime"]["home"],
            "away_goals": m["score"]["fullTime"]["away"],
            "status": m["status"],
        })

    df = pd.DataFrame(rows)
    # Garder uniquement les matchs terminés
    df = df[df["status"] == "FINISHED"].drop(columns=["status"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    return df


# ──────────────────────────────────────────────
# 2. Équipes participantes
# ──────────────────────────────────────────────

def scrape_ucl_teams(season: int = 2024) -> pd.DataFrame:
    """
    Récupère les équipes participantes à la UCL.
    Retourne : team_id, name, country, founded, venue
    """
    print(f"Scraping équipes UCL {season}...")
    data = _get("competitions/CL/teams", params={"season": season})

    rows = []
    for t in data.get("teams", []):
        rows.append({
            "team_id": t["id"],
            "name": t["name"],
            "short_name": t.get("shortName"),
            "country": t["area"]["name"],
            "founded": t.get("founded"),
            "venue": t.get("venue"),
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 3. Classement UCL (phase de ligue)
# ──────────────────────────────────────────────

def scrape_ucl_standings(season: int = 2024) -> pd.DataFrame:
    """
    Récupère le classement de la phase de ligue UCL.
    Retourne : position, team, played, won, draw, lost, goals_for, goals_against, points
    """
    print(f"Scraping classement UCL {season}...")
    try:
        data = _get("competitions/CL/standings", params={"season": season})
    except requests.exceptions.HTTPError as e:
        print(f"Classement indisponible ({e}) — calcul depuis les matchs à la place.")
        return pd.DataFrame()


    rows = []
    for group in data.get("standings", []):
        for entry in group.get("table", []):
            rows.append({
                "position": entry["position"],
                "team": entry["team"]["name"],
                "team_id": entry["team"]["id"],
                "played": entry["playedGames"],
                "won": entry["won"],
                "draw": entry["draw"],
                "lost": entry["lost"],
                "goals_for": entry["goalsFor"],
                "goals_against": entry["goalsAgainst"],
                "goal_diff": entry["goalDifference"],
                "points": entry["points"],
            })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 4. Historique UCL multi-saisons
# ──────────────────────────────────────────────

def scrape_ucl_history(seasons: list = None) -> pd.DataFrame:
    """
    Collecte les matchs UCL pour plusieurs saisons passées et les combine.
    Normalise les noms de phases pour unifier l'ancien et le nouveau format.
    Ajoute une colonne 'season' pour identifier chaque saison.
    """
    if seasons is None:
        seasons = HISTORICAL_SEASONS

    all_dfs = []
    for season in seasons:
        try:
            df = scrape_ucl_matches(season)
            df["season"] = season
            df["phase"] = df["phase"].replace(PHASE_NORMALIZE)
            all_dfs.append(df)
        except Exception as e:
            print(f"  Erreur saison {season}: {e}")
        time.sleep(1)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ──────────────────────────────────────────────
# 5. Classements ligues domestiques
# ──────────────────────────────────────────────

def scrape_domestic_standings(season: int = 2024) -> pd.DataFrame:
    """
    Récupère les classements finaux des 5 grands championnats européens.
    Utilisé comme proxy de la qualité globale d'une équipe.
    Les équipes hors top 5 (Ajax, Sporting, etc.) ne seront pas couvertes.
    """
    rows = []
    for code, league_name in DOMESTIC_LEAGUES.items():
        print(f"Scraping {league_name} ({code})...")
        try:
            data = _get(f"competitions/{code}/standings", params={"season": season})
        except Exception as e:
            print(f"  Indisponible ({e}), ignoré.")
            continue

        for group in data.get("standings", []):
            if group.get("type") != "TOTAL":
                continue
            for entry in group.get("table", []):
                rows.append({
                    "league":          code,
                    "league_name":     league_name,
                    "team_id":         entry["team"]["id"],
                    "team":            entry["team"]["name"],
                    "position":        entry["position"],
                    "played":          entry["playedGames"],
                    "won":             entry["won"],
                    "draw":            entry["draw"],
                    "lost":            entry["lost"],
                    "goals_for":       entry["goalsFor"],
                    "goals_against":   entry["goalsAgainst"],
                    "goal_diff":       entry["goalDifference"],
                    "points":          entry["points"],
                })
        time.sleep(1)  # sécurité rate limit entre chaque ligue

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 6. Sauvegarde
# ──────────────────────────────────────────────

def save_raw(df: pd.DataFrame, filename: str) -> None:
    """Sauvegarde un DataFrame en CSV dans data/raw/."""
    path = os.path.join(RAW_DATA_PATH, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Sauvegardé : {path} ({len(df)} lignes)")


# ──────────────────────────────────────────────
# 7. Pipeline principal
# ──────────────────────────────────────────────

if __name__ == "__main__":
    season = 2024

    print("\n=== Matchs UCL (saison courante) ===")
    matches = scrape_ucl_matches(season)
    matches["season"] = season
    matches["phase"] = matches["phase"].replace(PHASE_NORMALIZE)
    save_raw(matches, "ucl_matches.csv")

    print("\n=== Matchs UCL (historique 2019-2023) ===")
    history = scrape_ucl_history(HISTORICAL_SEASONS)
    all_matches = pd.concat([history, matches], ignore_index=True).sort_values("date")
    print(f"Total : {len(all_matches)} matchs sur {all_matches['season'].nunique()} saisons")
    save_raw(all_matches, "ucl_matches_all.csv")

    print("\n=== Équipes ===")
    teams = scrape_ucl_teams(season)
    save_raw(teams, "ucl_teams.csv")

    print("\n=== Classement UCL (saison courante) ===")
    standings = scrape_ucl_standings(season)
    save_raw(standings, "ucl_standings.csv")

    print("\n=== Classements ligues domestiques ===")
    domestic = scrape_domestic_standings(season)
    save_raw(domestic, "domestic_standings.csv")

    print("\nScraping terminé.")
