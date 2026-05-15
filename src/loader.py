"""
loader.py — Pont entre les CSVs bruts et la base DuckDB.

Charge les fichiers produits par scraper.py dans les tables DuckDB correspondantes.
Ce module est le point d'entrée unique vers la base de données : tous les autres
modules (features.py, model.py) passent par get_connection() pour lire les données.

Tables créées dans data/db/ucl.db :
  - raw_ucl_matches        : matchs UCL toutes saisons combinées
  - raw_ucl_standings      : classement UCL phase de ligue (saison courante)
  - raw_ucl_teams          : équipes participantes (saison courante)
  - raw_domestic_standings : classements des 5 grandes ligues (saison courante)

Note : ce module sera le point de migration vers dbt lors d'une future évolution
de l'architecture (les tables raw_ deviendront des sources dbt).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
from config.settings import DB_PATH, RAW_UCL_DIR, RAW_DOM_DIR


def get_connection() -> duckdb.DuckDBPyConnection:
    """Retourne une connexion à la base DuckDB du projet."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def _load_csv(conn, path: Path, table: str) -> int:
    """Charge un CSV dans une table DuckDB. Remplace la table si elle existe déjà."""
    if not path.exists():
        print(f"  ⚠  Introuvable : {path}")
        return 0
    df = pd.read_csv(path)
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.register("_tmp", df)
    conn.execute(f"CREATE TABLE {table} AS SELECT * FROM _tmp")
    conn.unregister("_tmp")
    print(f"  {table:<30} {len(df)} lignes")
    return len(df)


def load_ucl_matches(conn) -> int:
    """Préfère le fichier multi-saisons (all.csv) s'il existe."""
    all_path    = RAW_UCL_DIR / "matches" / "all.csv"
    single_path = RAW_UCL_DIR / "matches" / f"2024.csv"
    path = all_path if all_path.exists() else single_path
    return _load_csv(conn, path, "raw_ucl_matches")


def load_ucl_standings(conn) -> int:
    return _load_csv(conn, RAW_UCL_DIR / "standings" / "2024.csv", "raw_ucl_standings")


def load_ucl_teams(conn) -> int:
    return _load_csv(conn, RAW_UCL_DIR / "teams" / "2024.csv", "raw_ucl_teams")


def load_domestic_standings(conn) -> int:
    return _load_csv(conn, RAW_DOM_DIR / "2024.csv", "raw_domestic_standings")


def run() -> duckdb.DuckDBPyConnection:
    """Charge toutes les tables raw dans DuckDB et retourne la connexion."""
    conn = get_connection()
    print(f"Chargement dans DuckDB ({DB_PATH}) ...")
    load_ucl_matches(conn)
    load_ucl_standings(conn)
    load_ucl_teams(conn)
    load_domestic_standings(conn)
    print("Chargement terminé.\n")
    return conn


if __name__ == "__main__":
    run()
