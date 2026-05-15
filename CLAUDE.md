# CLAUDE.md — UCL Prediction

## Contexte

Projet de ML pour prédire le vainqueur de la UEFA Champions League à partir des statistiques des équipes. Données collectées via l'API `football-data.org`. Saison cible : 2024 (= saison 2024-2025).

## Structure des fichiers

```
UCL-prediction/
├── config/
│   └── settings.py          # Toutes les constantes et chemins du projet
├── data/
│   ├── raw/                 # CSVs bruts — gitignorés, produits par scraper.py
│   │   ├── ucl/
│   │   │   ├── matches/     # 2024.csv + all.csv (toutes saisons)
│   │   │   ├── standings/   # 2024.csv (classement phase de ligue)
│   │   │   └── teams/       # 2024.csv (équipes participantes)
│   │   └── domestic/        # 2024.csv (top 5 championnats)
│   └── db/
│       └── ucl.db           # Base DuckDB — gitignorée, produite par loader.py
├── models/
│   ├── best_model.pkl            # Meilleur modèle (features brutes)
│   └── best_model_normalized.pkl # Meilleur modèle (features normalisées)
├── src/
│   ├── scraper.py    # Collecte API → data/raw/
│   ├── loader.py     # Charge CSVs → DuckDB (tables raw_*)
│   ├── features.py   # Feature engineering → table 'features' dans DuckDB
│   └── model.py      # Entraînement, évaluation, simulation tournoi
├── main.py           # Orchestrateur — lance les 4 étapes en séquence
├── dashboard.html    # Dashboard interactif (données mock)
├── .env              # Non versionné — contient FOOTBALL_DATA_API_KEY
├── .env.example      # Template de .env — versionné
└── requirements.txt
```

## Variables d'environnement

Copier `.env.example` en `.env` et remplir :

```
FOOTBALL_DATA_API_KEY=ton_token_ici
```

Clé obtenue sur [football-data.org](https://www.football-data.org/) (tier gratuit disponible).

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env   # puis éditer avec ta clé API
```

## Ordre d'exécution

### Pipeline complet (recommandé)

```bash
python main.py
```

### Étapes individuelles

```bash
python src/scraper.py    # → data/raw/**/*.csv
python src/loader.py     # → data/db/ucl.db (tables raw_*)
python src/features.py   # → table 'features' dans DuckDB
python src/model.py      # → models/*.pkl + prédiction du vainqueur
```

### Sans re-scraper (données brutes déjà présentes)

```bash
python main.py --skip-scraping
```

## Tables DuckDB (ucl.db)

| Table | Contenu | Produite par |
|---|---|---|
| `raw_ucl_matches` | Matchs UCL toutes saisons | loader.py |
| `raw_ucl_standings` | Classement phase de ligue UCL | loader.py |
| `raw_ucl_teams` | Équipes participantes | loader.py |
| `raw_domestic_standings` | Classements top 5 ligues | loader.py |
| `features` | Dataset ML complet (37 features) | features.py |

## Features du modèle (37 au total)

- **Forme globale** (9) : stats rolling sur N derniers matchs UCL
- **Forme contextuelle** (9) : forme dom./ext. séparée + différentiels
- **Standings UCL** (9) : classement phase de ligue (knockout uniquement)
- **Stats domestique** (9) : classement championnat national
- **Phase encodée** (1) : ordinal 1-6

## Points d'attention

- **Pas de fuite de données** : la forme rolling est calculée sur les matchs PRÉCÉDANT le match courant.
- **Standings UCL** : appliqués uniquement aux phases knockout (classement non connu pendant la phase de ligue).
- **Fenêtre rolling adaptative** : 5 matchs en phase de ligue, 3 en knockout.
- **Convention nul** : dans `simulate_knockout`, un nul avantage l'équipe à domicile (arbitraire).
- **Bracket fixe** : `simulate_knockout` trie les équipes alphabétiquement.
- **Rate limit API** : le scraper attend 60s automatiquement si la limite est atteinte (429).

## Améliorations prévues

- Standings historiques (2019-2023) pour les saisons passées
- Stats head-to-head entre équipes
- `main.py --season 2023` pour rejouer une saison passée
- Migration vers dbt pour la couche de transformation (loader.py → dbt models)
