"""Carga el dataset procesado a Supabase (PostgreSQL).

Uso:
    python src/upload_to_supabase.py            # carga todo el processed.csv
    python src/upload_to_supabase.py --truncate # borra la tabla antes de cargar
    python src/upload_to_supabase.py --limit 10000   # carga solo las primeras N filas
    python src/upload_to_supabase.py --schema-only   # crea tabla e índices, no inserta

Variables de entorno (.env):
    DATABASE_URL: connection string Postgres de Supabase (Settings → Database
        → Connection string → URI). Usa el "Transaction pooler" (puerto 6543)
        para batch inserts.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_CSV = DATA_DIR / "processed.csv"

TABLE_NAME = "air_quality"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id               BIGSERIAL PRIMARY KEY,
    datetime         TIMESTAMP    NOT NULL,
    location_id      INTEGER      NOT NULL,
    name             TEXT,
    provider_name    TEXT,
    lat              NUMERIC(9, 6),
    lon              NUMERIC(9, 6),
    pm25             NUMERIC(10, 3),
    pm10             NUMERIC(10, 3),
    relativehumidity NUMERIC(10, 3),
    temperature      NUMERIC(10, 3),
    um003            NUMERIC(12, 3),
    hour             SMALLINT,
    day              SMALLINT,
    month            SMALLINT,
    year             SMALLINT,
    dayofweek        SMALLINT,
    aqi_class        SMALLINT     NOT NULL,
    aqi_label        TEXT         NOT NULL,
    CONSTRAINT uq_air_quality_loc_dt UNIQUE (location_id, datetime)
);

CREATE INDEX IF NOT EXISTS idx_air_quality_datetime  ON {TABLE_NAME} (datetime);
CREATE INDEX IF NOT EXISTS idx_air_quality_aqi_class ON {TABLE_NAME} (aqi_class);
CREATE INDEX IF NOT EXISTS idx_air_quality_location  ON {TABLE_NAME} (location_id);
"""

INSERT_SQL = f"""
INSERT INTO {TABLE_NAME} (
    datetime, location_id, name, provider_name, lat, lon,
    pm25, pm10, relativehumidity, temperature, um003,
    hour, day, month, year, dayofweek, aqi_class, aqi_label
) VALUES %s
ON CONFLICT (location_id, datetime) DO NOTHING
"""

COLUMNS_ORDER = [
    "datetime", "location_id", "name", "provider_name", "lat", "lon",
    "pm25", "pm10", "relativehumidity", "temperature", "um003",
    "hour", "day", "month", "year", "dayofweek", "aqi_class", "aqi_label",
]

BATCH_SIZE = 5000


def get_connection() -> psycopg2.extensions.connection:
    """Conecta a Postgres usando DATABASE_URL del .env."""
    load_dotenv(PROJECT_ROOT / ".env")
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        sys.exit(
            "ERROR: falta DATABASE_URL en .env. "
            "Tómala de Supabase → Settings → Database → Connection string (URI)."
        )
    return psycopg2.connect(db_url)


def ensure_schema(conn: psycopg2.extensions.connection) -> None:
    """Crea la tabla, constraint y índices si no existen."""
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    print(f"Tabla `{TABLE_NAME}` y constraints listos.")


def truncate_table(conn: psycopg2.extensions.connection) -> None:
    """Vacía la tabla (útil para reinicios limpios)."""
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {TABLE_NAME} RESTART IDENTITY;")
    conn.commit()
    print(f"Tabla `{TABLE_NAME}` vaciada.")


def load_dataframe(limit: int | None = None) -> pd.DataFrame:
    """Carga el CSV procesado en el orden de columnas correcto."""
    if not PROCESSED_CSV.exists():
        sys.exit(
            f"ERROR: no existe {PROCESSED_CSV}. "
            "Corre primero el notebook 01_eda.ipynb para generarlo."
        )
    df = pd.read_csv(PROCESSED_CSV, parse_dates=["datetime"])
    if limit:
        df = df.head(limit)
    df = df[COLUMNS_ORDER]
    # psycopg2 maneja NaN/NaT mal — los convertimos a None.
    df = df.astype(object).where(pd.notna(df), None)
    return df


def insert_dataframe(
    conn: psycopg2.extensions.connection, df: pd.DataFrame
) -> None:
    """Inserta el DataFrame en batches con ON CONFLICT DO NOTHING."""
    total = len(df)
    rows = df.itertuples(index=False, name=None)
    inserted = 0

    with conn.cursor() as cur, tqdm(total=total, desc="Insertando") as pbar:
        batch: list[tuple] = []
        for row in rows:
            batch.append(row)
            if len(batch) >= BATCH_SIZE:
                execute_values(cur, INSERT_SQL, batch, page_size=BATCH_SIZE)
                conn.commit()
                inserted += len(batch)
                pbar.update(len(batch))
                batch = []
        if batch:
            execute_values(cur, INSERT_SQL, batch, page_size=BATCH_SIZE)
            conn.commit()
            inserted += len(batch)
            pbar.update(len(batch))

    print(f"Filas procesadas: {inserted} (duplicados por (location_id, datetime) se ignoraron)")


def count_rows(conn: psycopg2.extensions.connection) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
        return cur.fetchone()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--truncate", action="store_true",
        help="Vacía la tabla antes de insertar",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Inserta solo las primeras N filas (útil para probar)",
    )
    parser.add_argument(
        "--schema-only", action="store_true",
        help="Solo crea la tabla e índices; no inserta datos",
    )
    args = parser.parse_args()

    conn = get_connection()
    try:
        ensure_schema(conn)
        if args.schema_only:
            print("Modo schema-only: tabla creada, no se insertaron datos.")
            return

        if args.truncate:
            truncate_table(conn)

        df = load_dataframe(limit=args.limit)
        print(f"Cargando {len(df):,} filas a `{TABLE_NAME}` (batches de {BATCH_SIZE})...")
        insert_dataframe(conn, df)

        n = count_rows(conn)
        print(f"\nTotal de filas ahora en `{TABLE_NAME}`: {n:,}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
