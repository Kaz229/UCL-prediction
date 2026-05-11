"""
model.py — Entraînement, évaluation et prédiction UCL
Deux tâches :
  1. Prédiction du résultat d'un match (win/draw/loss)
  2. Simulation du tournoi pour prédire le vainqueur
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")

FEATURE_COLS = [
    # Forme globale (tous matchs UCL)
    "home_form_gf", "home_form_ga", "home_form_pts",
    "away_form_gf", "away_form_ga", "away_form_pts",
    "diff_form_pts", "diff_form_gf", "diff_form_ga",
    # Forme contextuelle (dom. pour l'équipe domicile, ext. pour l'équipe visiteuse)
    "home_form_at_home_gf", "home_form_at_home_ga", "home_form_at_home_pts",
    "away_form_at_away_gf", "away_form_at_away_ga", "away_form_at_away_pts",
    "diff_form_ctx_pts", "diff_form_ctx_gf", "diff_form_ctx_ga",
    # Classement phase de ligue (knockout uniquement)
    "home_standing_position", "home_standing_points", "home_standing_goal_diff",
    "away_standing_position", "away_standing_points", "away_standing_goal_diff",
    "diff_standing_pts", "diff_standing_pos", "diff_standing_gd",
    "phase_encoded",
]
TARGET = "result"

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
}


# ──────────────────────────────────────────────
# 1. Chargement des données
# ──────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    path = os.path.join(PROCESSED_PATH, "ucl_features.csv")
    return pd.read_csv(path, parse_dates=["date"])


def split(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ──────────────────────────────────────────────
# 2. Entraînement & évaluation (version normale)
# ──────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame, normalize: bool = False) -> dict:
    """
    Entraîne et évalue tous les modèles.
    normalize=True  → StandardScaler sur les features (version normalisée)
    normalize=False → features brutes (version normale)
    Retourne les métriques et les modèles entraînés.
    """
    label = "normalisée" if normalize else "normale"
    print(f"\n{'='*50}")
    print(f"  Version {label}")
    print(f"{'='*50}")

    X_train, X_test, y_train, y_test = split(df)

    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    results = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Cross-validation sur tout le dataset
        X_all = df[FEATURE_COLS]
        if normalize and scaler:
            X_all = scaler.transform(X_all)
        cv_scores = cross_val_score(model, X_all, df[TARGET], cv=5, scoring="accuracy")

        print(f"\n{name}")
        print(f"  Accuracy test  : {acc:.4f}")
        print(f"  CV accuracy    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(classification_report(y_test, y_pred,
                                    target_names=["Défaite dom.", "Nul", "Victoire dom."],
                                    zero_division=0))

        results[name] = {
            "model": model,
            "scaler": scaler,
            "accuracy": acc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

    return results


# ──────────────────────────────────────────────
# 3. Sauvegarde du meilleur modèle
# ──────────────────────────────────────────────

def save_best(results: dict, suffix: str = "") -> None:
    """Sauvegarde le modèle avec la meilleure CV accuracy."""
    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    best = results[best_name]
    os.makedirs(MODELS_PATH, exist_ok=True)

    filename = f"best_model{suffix}.pkl"
    path = os.path.join(MODELS_PATH, filename)
    with open(path, "wb") as f:
        pickle.dump({"name": best_name, "model": best["model"], "scaler": best["scaler"]}, f)

    print(f"\nMeilleur modèle ({suffix or 'normal'}) : {best_name}")
    print(f"  CV accuracy : {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
    print(f"  Sauvegardé  : {path}")


# ──────────────────────────────────────────────
# 4. Simulation du tournoi (prédiction du vainqueur)
# ──────────────────────────────────────────────

def _team_form(df: pd.DataFrame, team: str) -> pd.Series:
    """Récupère les dernières features de forme d'une équipe."""
    home = df[df["home_team"] == team][["home_form_gf", "home_form_ga", "home_form_pts"]].tail(1)
    away = df[df["away_team"] == team][["away_form_gf", "away_form_ga", "away_form_pts"]].tail(1)

    if not home.empty:
        row = home.iloc[0]
        return pd.Series({
            "form_gf": row["home_form_gf"],
            "form_ga": row["home_form_ga"],
            "form_pts": row["home_form_pts"],
        })
    elif not away.empty:
        row = away.iloc[0]
        return pd.Series({
            "form_gf": row["away_form_gf"],
            "form_ga": row["away_form_ga"],
            "form_pts": row["away_form_pts"],
        })
    return pd.Series({"form_gf": 0, "form_ga": 0, "form_pts": 0})


def predict_match(model, scaler, home_form: dict, away_form: dict, phase: int) -> int:
    """Prédit le résultat d'un match à partir des formes des deux équipes."""
    features = pd.DataFrame([{
        "home_form_gf": home_form["form_gf"],
        "home_form_ga": home_form["form_ga"],
        "home_form_pts": home_form["form_pts"],
        "away_form_gf": away_form["form_gf"],
        "away_form_ga": away_form["form_ga"],
        "away_form_pts": away_form["form_pts"],
        "diff_form_pts": home_form["form_pts"] - away_form["form_pts"],
        "diff_form_gf": home_form["form_gf"] - away_form["form_gf"],
        "diff_form_ga": home_form["form_ga"] - away_form["form_ga"],
        "phase_encoded": phase,
    }])
    if scaler:
        features = scaler.transform(features)
    return model.predict(features)[0]


def simulate_knockout(df: pd.DataFrame, model, scaler, teams: list, phase: int) -> str:
    """
    Simule une phase à élimination directe entre les équipes données.
    Retourne le nom du vainqueur.
    """
    remaining = teams[:]
    round_num = 0

    while len(remaining) > 1:
        round_num += 1
        next_round = []
        pairs = [(remaining[i], remaining[i+1]) for i in range(0, len(remaining), 2)]

        for home, away in pairs:
            home_form = _team_form(df, home).to_dict()
            away_form = _team_form(df, away).to_dict()
            result = predict_match(model, scaler, home_form, away_form, phase)
            # En cas de nul, on avantage légèrement l'équipe à domicile (convention)
            winner = home if result >= 1 else away
            next_round.append(winner)

        remaining = next_round
        phase = min(phase + 1, 6)

    return remaining[0]


def predict_winner(df: pd.DataFrame, model_path: str = None) -> None:
    """
    Charge le meilleur modèle et simule le tournoi pour prédire le vainqueur.
    """
    if model_path is None:
        model_path = os.path.join(MODELS_PATH, "best_model.pkl")

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
    scaler = saved["scaler"]
    name = saved["name"]
    print(f"\nModèle utilisé : {name}")

    # Équipes en lice (toutes les équipes du dataset)
    teams = sorted(set(df["home_team"].tolist() + df["away_team"].tolist()))
    # Aligner sur une puissance de 2 (prendre les 32 premières)
    teams = teams[:32]

    print(f"Équipes simulées : {len(teams)}")
    winner = simulate_knockout(df, model, scaler, teams, phase=3)
    print(f"\nVainqueur prédit de la UCL : {winner}")


# ──────────────────────────────────────────────
# 5. Pipeline principal
# ──────────────────────────────────────────────

if __name__ == "__main__":
    df = load_features()
    print(f"Dataset chargé : {df.shape[0]} matchs, {len(FEATURE_COLS)} features")

    # Version normale
    results_normal = train_and_evaluate(df, normalize=False)
    save_best(results_normal, suffix="")

    # Version normalisée
    results_norm = train_and_evaluate(df, normalize=True)
    save_best(results_norm, suffix="_normalized")

    # Prédiction du vainqueur
    print("\n" + "="*50)
    print("  Simulation du tournoi")
    print("="*50)
    predict_winner(df)
