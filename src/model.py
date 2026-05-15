"""
model.py — Entraînement, évaluation et simulation du tournoi UCL.

Lit les features depuis DuckDB (table 'features'), compare trois modèles
de classification et sauvegarde le meilleur sur disque.

Deux tâches :
  1. Prédiction du résultat d'un match (victoire dom. / nul / défaite dom.)
  2. Simulation d'un bracket knockout pour prédire le vainqueur de la UCL

Modèles comparés : Logistic Regression, Random Forest, Gradient Boosting
Évaluation       : accuracy + cross-validation 5 folds
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.loader import get_connection
from config.settings import FEATURE_COLS, TARGET, MODELS_DIR

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
}


# ──────────────────────────────────────────────
# 1. Chargement des données
# ──────────────────────────────────────────────

def load_features(conn=None) -> pd.DataFrame:
    """Charge le dataset de features depuis la table DuckDB 'features'."""
    if conn is None:
        conn = get_connection()
    return conn.execute("SELECT * FROM features ORDER BY date").df()


def split(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ──────────────────────────────────────────────
# 2. Entraînement & évaluation
# ──────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame, normalize: bool = False) -> dict:
    """
    Entraîne et évalue tous les modèles.
    normalize=True  → StandardScaler appliqué sur les features
    normalize=False → features brutes
    """
    label = "normalisée" if normalize else "normale"
    print(f"\n{'='*50}\n  Version {label}\n{'='*50}")

    X_train, X_test, y_train, y_test = split(df)

    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    results = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)

        X_all = scaler.transform(df[FEATURE_COLS]) if normalize and scaler else df[FEATURE_COLS]
        cv    = cross_val_score(model, X_all, df[TARGET], cv=5, scoring="accuracy")

        print(f"\n{name}")
        print(f"  Accuracy test : {acc:.4f}")
        print(f"  CV accuracy   : {cv.mean():.4f} ± {cv.std():.4f}")
        print(classification_report(y_test, y_pred,
                                    target_names=["Défaite dom.", "Nul", "Victoire dom."],
                                    zero_division=0))

        results[name] = {"model": model, "scaler": scaler, "accuracy": acc,
                         "cv_mean": cv.mean(), "cv_std": cv.std()}
    return results


# ──────────────────────────────────────────────
# 3. Sauvegarde du meilleur modèle
# ──────────────────────────────────────────────

def save_best(results: dict, suffix: str = "") -> None:
    """Sauvegarde sur disque le modèle avec la meilleure CV accuracy."""
    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    best = results[best_name]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    path = MODELS_DIR / f"best_model{suffix}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"name": best_name, "model": best["model"], "scaler": best["scaler"]}, f)

    print(f"\nMeilleur modèle ({suffix or 'normal'}) : {best_name}")
    print(f"  CV accuracy : {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
    print(f"  Sauvegardé  : {path}")


# ──────────────────────────────────────────────
# 4. Simulation du tournoi
# ──────────────────────────────────────────────

def _team_form(df: pd.DataFrame, team: str) -> dict:
    """Récupère les dernières features de forme d'une équipe."""
    home = df[df["home_team"] == team][["home_form_gf", "home_form_ga", "home_form_pts"]].tail(1)
    away = df[df["away_team"] == team][["away_form_gf", "away_form_ga", "away_form_pts"]].tail(1)

    if not home.empty:
        r = home.iloc[0]
        return {"form_gf": r["home_form_gf"], "form_ga": r["home_form_ga"], "form_pts": r["home_form_pts"]}
    if not away.empty:
        r = away.iloc[0]
        return {"form_gf": r["away_form_gf"], "form_ga": r["away_form_ga"], "form_pts": r["away_form_pts"]}
    return {"form_gf": 0, "form_ga": 0, "form_pts": 0}


def predict_match(model, scaler, home_form: dict, away_form: dict, phase: int) -> int:
    """Prédit le résultat d'un match à partir des formes des deux équipes."""
    # Construit un vecteur de features avec les colonnes attendues par le modèle.
    # Les features non disponibles ici (standings, domestic) sont laissées à 0.
    row = {col: 0 for col in FEATURE_COLS}
    row.update({
        "home_form_gf":  home_form["form_gf"],
        "home_form_ga":  home_form["form_ga"],
        "home_form_pts": home_form["form_pts"],
        "away_form_gf":  away_form["form_gf"],
        "away_form_ga":  away_form["form_ga"],
        "away_form_pts": away_form["form_pts"],
        "diff_form_pts": home_form["form_pts"] - away_form["form_pts"],
        "diff_form_gf":  home_form["form_gf"]  - away_form["form_gf"],
        "diff_form_ga":  home_form["form_ga"]  - away_form["form_ga"],
        "phase_encoded": phase,
    })
    features = pd.DataFrame([row])[FEATURE_COLS]
    if scaler:
        features = scaler.transform(features)
    return model.predict(features)[0]


def simulate_knockout(df: pd.DataFrame, model, scaler, teams: list, phase: int) -> str:
    """
    Simule un bracket à élimination directe et retourne le vainqueur prédit.
    Convention : un match nul avantage l'équipe à domicile (arbitraire).
    """
    remaining = teams[:]
    while len(remaining) > 1:
        next_round = []
        for i in range(0, len(remaining), 2):
            home, away = remaining[i], remaining[i + 1]
            result = predict_match(model, scaler,
                                   _team_form(df, home),
                                   _team_form(df, away), phase)
            next_round.append(home if result >= 1 else away)
        remaining = next_round
        phase = min(phase + 1, 6)
    return remaining[0]


def predict_winner(df: pd.DataFrame, model_path: Path = None) -> str:
    """Charge le meilleur modèle et prédit le vainqueur via simulation knockout."""
    if model_path is None:
        model_path = MODELS_DIR / "best_model.pkl"

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    model, scaler, name = saved["model"], saved["scaler"], saved["name"]
    print(f"\nModèle utilisé : {name}")

    teams = sorted(set(df["home_team"].tolist() + df["away_team"].tolist()))[:32]
    print(f"Équipes simulées : {len(teams)}")
    winner = simulate_knockout(df, model, scaler, teams, phase=3)
    print(f"\nVainqueur prédit de la UCL : {winner}")
    return winner


# ──────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────

def run(conn=None) -> None:
    """Charge les features depuis DuckDB, entraîne les modèles et prédit le vainqueur."""
    if conn is None:
        conn = get_connection()

    df = load_features(conn)
    print(f"[Modèle] Dataset : {df.shape[0]} matchs, {len(FEATURE_COLS)} features")

    results_normal = train_and_evaluate(df, normalize=False)
    save_best(results_normal, suffix="")

    results_norm = train_and_evaluate(df, normalize=True)
    save_best(results_norm, suffix="_normalized")

    print("\n" + "=" * 50 + "\n  Simulation du tournoi\n" + "=" * 50)
    predict_winner(df)
    print("\n[Modèle] Terminé.")


if __name__ == "__main__":
    run()
