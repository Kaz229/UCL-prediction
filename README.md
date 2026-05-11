# UCL Prediction

Projet de machine learning pour prédire le vainqueur de la UEFA Champions League à partir des statistiques des équipes.

## Stack

- **Python 3.10+**
- **scikit-learn** — Logistic Regression, Random Forest, Gradient Boosting
- **pandas** — manipulation des données
- **football-data.org API** — source des données (matchs, classements)

## Architecture

```
football-data.org API
        ↓
   scraper.py  →  data/raw/         (matchs, équipes, classement)
        ↓
   features.py →  data/processed/   (forme récente, features différentielles)
        ↓
   model.py    →  models/           (modèle entraîné + prédiction du vainqueur)
```

## Installation

```bash
git clone https://github.com/kaz229/ucl-prediction.git
cd ucl-prediction
pip install -r requirements.txt
```

Créer un fichier `.env` à la racine :

```
FOOTBALL_DATA_API_KEY=ton_token_ici
```

Clé disponible gratuitement sur [football-data.org](https://www.football-data.org/).

## Usage

Les 3 étapes sont à exécuter dans l'ordre :

```bash
# 1. Collecte des données
python src/scraper.py

# 2. Feature engineering
python src/features.py

# 3. Entraînement + prédiction du vainqueur
python src/model.py
```

## Features du modèle

Les features sont calculées sur les 5 derniers matchs UCL de chaque équipe, **avant** le match à prédire :

| Feature | Description |
|---|---|
| `home_form_gf` | Buts marqués (domicile, 5 derniers matchs) |
| `home_form_ga` | Buts concédés (domicile) |
| `home_form_pts` | Points (domicile) |
| `away_form_gf` | Buts marqués (extérieur) |
| `away_form_ga` | Buts concédés (extérieur) |
| `away_form_pts` | Points (extérieur) |
| `diff_form_pts` | Écart de points (home - away) |
| `diff_form_gf` | Écart de buts marqués |
| `diff_form_ga` | Écart de buts concédés |
| `phase_encoded` | Phase du tournoi (1 = Ligue → 6 = Finale) |

**Variable cible** : résultat du match — `2` victoire domicile, `1` nul, `0` défaite domicile.

## Modèles comparés

Trois modèles sont entraînés et évalués en cross-validation (5 folds) :

- Logistic Regression
- Random Forest (200 estimateurs)
- **Gradient Boosting (200 estimateurs)** ← généralement meilleur

Deux versions sont produites : features brutes et features normalisées (StandardScaler).

## Roadmap

- [ ] Script `main.py` pour orchestrer les 3 étapes
- [ ] Intégration des stats de championnat domestic (pas seulement UCL)
- [ ] Simulation knockout avec tirage au sort réel
- [ ] Prise en compte des matchs aller-retour
- [ ] Passage à DuckDB + dbt pour la couche de transformation
