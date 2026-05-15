"""
scraper.py — Collecte des données UCL et des ligues domestiques via football-data.org.

Produit les fichiers CSV bruts dans data/raw/ :
  - data/raw/ucl/matches/2024.csv   : matchs UCL saison courante
  - data/raw/ucl/matches/all.csv    : matchs UCL toutes saisons (2019-2024)
  - data/raw/ucl/standings/2024.csv : classement phase de ligue UCL
  - data/raw/ucl/teams/2024.csv     : équipes participantes
  - data/raw/domestic/2024.csv      : classements des 5 grands championnats

Ces fichiers sont ensuite chargés dans DuckDB par loader.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import requests
import pandas as pd
from config.settings import (
    FOOTBALL_DATA_API_KEY, BASE_URL,
    CURRENT_SEASON, HISTORICAL_SEASONS,
    DOMESTIC_LEAGUES, PHASE_NORMALIZE,
    RAW_UCL_DIR, RAW_DOM_DIR,
)

HEADERS = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}


# ──────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────

def _get(endpoint: str, params: dict = None) -> dict:
    """GET sur l'API avec gestion automatique du rate limit (429)."""
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 429:
        print("  Rate limit atteint, attente 60s...")
        time.sleep(60)
        response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json()


def _save(df: pd.DataFrame, path: Path) -> None:
    """Sauvegarde un DataFrame en CSV, crée les dossiers si nécessaire."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Sauvegardé : {path} ({len(df)} lignes)")


# ──────────────────────────────────────────────
# 1. Matchs UCL
# ──────────────────────────────────────────────

def scrape_ucl_matches(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    Récupère tous les matchs UCL terminés d'une saison.
    season=2024 correspond à la saison 2024-2025.
    """
    print(f"  Scraping matchs UCL {season}...")
    data = _get("competitions/CL/matches", params={"season": season})

    rows = []
    for m in data.get("matches", []):
        rows.append({
            "match_id":   m["id"],
            "date":       m["utcDate"][:10],
            "phase":      m["stage"],
            "matchday":   m.get("matchday"),
            "home_team":  m["homeTeam"]["name"],
            "away_team":  m["awayTeam"]["name"],
            "home_goals": m["score"]["fullTime"]["home"],
            "away_goals": m["score"]["fullTime"]["away"],
            "status":     m["status"],
        })

    df = pd.DataFrame(rows)
    df = df[df["status"] == "FINISHED"].drop(columns=["status"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    return df


# ──────────────────────────────────────────────
# 2. Historique multi-saisons
# ──────────────────────────────────────────────

def scrape_ucl_history(seasons: list = None) -> pd.DataFrame:
    """
    Collecte les matchs UCL pour plusieurs saisons et les combine.
    Normalise les noms de phases pour unifier l'ancien et le nouveau format UCL.
    Ajoute une colonne 'season' sur chaque ligne.
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
# 3. Équipes participantes
# ──────────────────────────────────────────────

def scrape_ucl_teams(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Récupère les équipes participantes à la UCL pour une saison donnée."""
    print(f"  Scraping équipes UCL {season}...")
    data = _get("competitions/CL/teams", params={"season": season})

    rows = []
    for t in data.get("teams", []):
        rows.append({
            "team_id":    t["id"],
            "name":       t["name"],
            "short_name": t.get("shortName"),
            "country":    t["area"]["name"],
            "founded":    t.get("founded"),
            "venue":      t.get("venue"),
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 4. Classement UCL (phase de ligue)
# ──────────────────────────────────────────────

def scrape_ucl_standings(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    Récupère le classement de la phase de ligue UCL.
    Retourne un DataFrame vide si le classement est indisponible.
    """
    print(f"  Scraping classement UCL {season}...")
    try:
        data = _get("competitions/CL/standings", params={"season": season})
    except requests.exceptions.HTTPError as e:
        print(f"  Classement indisponible ({e})")
        return pd.DataFrame()

    rows = []
    for group in data.get("standings", []):
        for entry in group.get("table", []):
            rows.append({
                "position":      entry["position"],
                "team":          entry["team"]["name"],
                "team_id":       entry["team"]["id"],
                "played":        entry["playedGames"],
                "won":           entry["won"],
                "draw":          entry["draw"],
                "lost":          entry["lost"],
                "goals_for":     entry["goalsFor"],
                "goals_against": entry["goalsAgainst"],
                "goal_diff":     entry["goalDifference"],
                "points":        entry["points"],
            })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 5. Classements ligues domestiques
# ──────────────────────────────────────────────

def scrape_domestic_standings(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    Récupère les classements finaux des 5 grands championnats (PL, Liga, BL1, Serie A, L1).
    Utilisé comme proxy de la qualité d'une équipe dans son championnat national.
    Les équipes hors top 5 (Ajax, Sporting, etc.) ne seront pas couvertes.
    """
    rows = []
    for code, league_name in DOMESTIC_LEAGUES.items():
        print(f"  Scraping {league_name} ({code})...")
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
                    "league":        code,
                    "league_name":   league_name,
                    "team_id":       entry["team"]["id"],
                    "team":          entry["team"]["name"],
                    "position":      entry["position"],
                    "played":        entry["playedGames"],
                    "won":           entry["won"],
                    "draw":          entry["draw"],
                    "lost":          entry["lost"],
                    "goals_for":     entry["goalsFor"],
                    "goals_against": entry["goalsAgainst"],
                    "goal_diff":     entry["goalDifference"],
                    "points":        entry["points"],
                })
        time.sleep(1)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 6. Pipeline principal
# ──────────────────────────────────────────────

def run(season: int = CURRENT_SEASON) -> None:
    """Lance le scraping complet et sauvegarde tous les fichiers bruts."""

    print("\n[Scraper] Matchs UCL saison courante")
    matches = scrape_ucl_matches(season)
    matches["season"] = season
    matches["phase"] = matches["phase"].replace(PHASE_NORMALIZE)
    _save(matches, RAW_UCL_DIR / "matches" / f"{season}.csv")

    print("\n[Scraper] Historique UCL (saisons passées)")
    history = scrape_ucl_history(HISTORICAL_SEASONS)
    all_matches = pd.concat([history, matches], ignore_index=True).sort_values("date")
    print(f"  Total : {len(all_matches)} matchs sur {all_matches['season'].nunique()} saisons")
    _save(all_matches, RAW_UCL_DIR / "matches" / "all.csv")

    print("\n[Scraper] Équipes UCL")
    teams = scrape_ucl_teams(season)
    _save(teams, RAW_UCL_DIR / "teams" / f"{season}.csv")

    print("\n[Scraper] Classement UCL")
    standings = scrape_ucl_standings(season)
    _save(standings, RAW_UCL_DIR / "standings" / f"{season}.csv")

    print("\n[Scraper] Classements ligues domestiques")
    domestic = scrape_domestic_standings(season)
    _save(domestic, RAW_DOM_DIR / f"{season}.csv")

    print("\n[Scraper] Terminé.")


if __name__ == "__main__":
    run()
