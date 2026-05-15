"""
settings.py — Configuration centrale du projet UCL Prediction.

Toutes les constantes du projet sont définies ici : chemins, paramètres API,
saisons, phases, features. Les autres modules importent depuis ce fichier
plutôt que de définir leurs propres constantes.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Chemins
# ──────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent.parent

DATA_DIR     = ROOT_DIR / "data"
RAW_DIR      = DATA_DIR / "raw"
RAW_UCL_DIR  = RAW_DIR  / "ucl"
RAW_DOM_DIR  = RAW_DIR  / "domestic"
DB_PATH      = DATA_DIR / "db" / "ucl.db"
MODELS_DIR   = ROOT_DIR / "models"

# ──────────────────────────────────────────────
# API
# ──────────────────────────────────────────────

FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4"

# ──────────────────────────────────────────────
# Saisons
# ──────────────────────────────────────────────

CURRENT_SEASON    = 2024
HISTORICAL_SEASONS = [2019, 2020, 2021, 2022, 2023]
ALL_SEASONS       = HISTORICAL_SEASONS + [CURRENT_SEASON]

# ──────────────────────────────────────────────
# Ligues domestiques (tier gratuit API)
# ──────────────────────────────────────────────

DOMESTIC_LEAGUES = {
    "PL":  "Premier League",
    "PD":  "Primera Division",
    "BL1": "Bundesliga",
    "SA":  "Serie A",
    "FL1": "Ligue 1",
}

# ──────────────────────────────────────────────
# Phases UCL
# ──────────────────────────────────────────────

# L'UCL a changé de format en 2024 : GROUP_STAGE → LEAGUE_STAGE, ROUND_OF_16 → LAST_16
PHASE_NORMALIZE = {
    "GROUP_STAGE": "LEAGUE_STAGE",
    "ROUND_OF_16": "LAST_16",
}

# Ordre ordinal des phases (utilisé comme feature)
PHASE_ORDER = {
    "LEAGUE_STAGE":   1,
    "PLAYOFFS":       2,
    "LAST_16":        3,
    "QUARTER_FINALS": 4,
    "SEMI_FINALS":    5,
    "FINAL":          6,
}

# Phases après la phase de ligue (standings UCL disponibles sans fuite de données)
KNOCKOUT_PHASES = {"PLAYOFFS", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL"}

# ──────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────

# Fenêtre rolling adaptée : 5 matchs en phase de ligue, 3 en knockout
# (en knockout, les 3 derniers matchs reflètent mieux la forme récente)
PHASE_WINDOW = {
    "LEAGUE_STAGE":   5,
    "PLAYOFFS":       3,
    "LAST_16":        3,
    "QUARTER_FINALS": 3,
    "SEMI_FINALS":    3,
    "FINAL":          3,
}

FEATURE_COLS = [
    # Forme globale (tous matchs UCL, fenêtre adaptative)
    "home_form_gf", "home_form_ga", "home_form_pts",
    "away_form_gf", "away_form_ga", "away_form_pts",
    "diff_form_pts", "diff_form_gf", "diff_form_ga",
    # Forme contextuelle (domicile pour l'équipe home, extérieur pour l'équipe away)
    "home_form_at_home_gf", "home_form_at_home_ga", "home_form_at_home_pts",
    "away_form_at_away_gf", "away_form_at_away_ga", "away_form_at_away_pts",
    "diff_form_ctx_pts", "diff_form_ctx_gf", "diff_form_ctx_ga",
    # Classement phase de ligue UCL (knockout uniquement — pas de fuite)
    "home_standing_position", "home_standing_points", "home_standing_goal_diff",
    "away_standing_position", "away_standing_points", "away_standing_goal_diff",
    "diff_standing_pts", "diff_standing_pos", "diff_standing_gd",
    # Classement ligue domestique (top 5 championnats, toutes phases)
    "home_domestic_position", "home_domestic_points", "home_domestic_goal_diff",
    "away_domestic_position", "away_domestic_points", "away_domestic_goal_diff",
    "diff_domestic_pts", "diff_domestic_pos", "diff_domestic_gd",
    # Phase du tournoi
    "phase_encoded",
]

TARGET = "result"
