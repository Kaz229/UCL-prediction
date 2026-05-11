# CLAUDE.md — UCL Prediction

## Contexte

Projet de ML pour prédire le vainqueur de la UEFA Champions League à partir des statistiques des équipes. Données collectées via l'API `football-data.org`. Saison cible : 2024 (= saison 2024-2025).

## Structure des fichiers

```
UCL-prediction/
├── src/
│   ├── scraper.py    # Collecte des données via API (matchs, équipes, classement)
│   ├── features.py   # Feature engineering à partir des données brutes
│   └── model.py      # Entraînement, évaluation et simulation du tournoi
├── data/
│   ├── raw/          # CSVs bruts produits par scraper.py
│   └── processed/    # CSVs transformés produits par features.py
├── models/
│   ├── best_model.pkl            # Meilleur modèle (features brutes)
│   └── best_model_normalized.pkl # Meilleur modèle (features normalisées)
├── requirements.txt
└── .env              # Non versionné — contient FOOTBALL_DATA_API_KEY
```

## Variables d'environnement

Créer un fichier `.env` à la racine :

```
FOOTBALL_DATA_API_KEY=ton_token_ici
```

Clé obtenue sur [football-data.org](https://www.football-data.org/) (tier gratuit disponible).

## Ordre d'exécution

Les 3 étapes sont séquentielles et dépendantes :

```bash
python src/scraper.py    # → data/raw/*.csv
python src/features.py   # → data/processed/ucl_features.csv
python src/model.py      # → models/best_model*.pkl + prédiction du vainqueur
```

## Points d'attention

- **Pas de fuite de données** : dans `features.py`, la forme rolling est calculée sur les matchs *précédant* le match courant. L'historique est mis à jour après chaque itération.
- **Convention nul** : dans la simulation knockout (`simulate_knockout`), un match nul avantage l'équipe à domicile. C'est une convention arbitraire, pas une règle officielle.
- **Bracket fixe** : `simulate_knockout` utilise les équipes triées alphabétiquement, pas un tirage au sort réel.
- **Rate limit API** : le scraper attend 60s automatiquement si la limite est atteinte (429).

## Améliorations connues

- Pas de script `main.py` pour orchestrer les 3 étapes en une seule commande
- Pas de données de saisons passées pour évaluer la précision historique du modèle
- Les features sont limitées à la forme UCL — pas de stats de championnat domestic
- La simulation knockout ignore les matchs aller-retour (deux matchs par confrontation en réalité)
