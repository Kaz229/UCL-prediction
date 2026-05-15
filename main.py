"""
main.py — Orchestrateur du pipeline UCL Prediction.

Lance les 4 étapes dans l'ordre :
  1. Scraping  : collecte les données via l'API football-data.org → data/raw/
  2. Loader    : charge les CSVs bruts dans DuckDB → data/db/ucl.db
  3. Features  : feature engineering → table 'features' dans DuckDB
  4. Modèle    : entraînement + prédiction du vainqueur → models/*.pkl

Usage :
  python main.py                  # pipeline complet
  python main.py --skip-scraping  # si les données brutes existent déjà
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src import scraper, loader, features, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline UCL Prediction")
    parser.add_argument(
        "--skip-scraping",
        action="store_true",
        help="Ignorer le scraping API (utiliser les données brutes existantes)",
    )
    args = parser.parse_args()

    if not args.skip_scraping:
        print("\n" + "=" * 60)
        print("  ÉTAPE 1/4 — Scraping")
        print("=" * 60)
        scraper.run()
    else:
        print("\n[1/4] Scraping ignoré (--skip-scraping)")

    print("\n" + "=" * 60)
    print("  ÉTAPE 2/4 — Chargement dans DuckDB")
    print("=" * 60)
    conn = loader.run()

    print("\n" + "=" * 60)
    print("  ÉTAPE 3/4 — Feature engineering")
    print("=" * 60)
    features.run(conn)

    print("\n" + "=" * 60)
    print("  ÉTAPE 4/4 — Entraînement du modèle")
    print("=" * 60)
    model.run(conn)

    print("\n" + "=" * 60)
    print("  Pipeline terminé.")
    print("=" * 60)


if __name__ == "__main__":
    main()
