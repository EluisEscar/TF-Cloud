"""Sube el Parquet multinacional generado por prepare_data.py a Supabase.

La tabla air_quality del MVP original solo tenía las mediciones de Almaty.
Este script añade las del modelo nuevo (10 países OpenAQ) para que la tab
"Histórico" del frontend muestre datos consistentes con el modelo en EC2.

Por defecto **hace APPEND** (los datos viejos de Almaty siguen). Pasa
``--truncate`` para empezar con una tabla limpia.

Uso:
    python ml_pipeline/upload_to_supabase.py
    python ml_pipeline/upload_to_supabase.py --truncate

Requisitos:
    pip install psycopg2-binary pandas pyarrow python-dotenv
    Y en .env: DATABASE_URL (Supabase → Settings → Database → Connection string)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = PROJECT_ROOT / "data" / "openaq_processed.parquet"
TABLE_NAME = "air_quality"
BATCH_SIZE = 5000  # Session pooler aguanta esto sin drop; menos overhead de
                   # reconexión que batches chicos. Reduce a 1000 si ves SSL drops.
COMMIT_EVERY = 5   # log de progreso cada N batches
MAX_RETRIES = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("upload")

# DDL idempotente: crea la tabla desde cero si no existe (proyecto Supabase
# nuevo) y aplica ALTER por si el proyecto tiene una versión vieja del
# schema sin country_code.
ENSURE_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id               BIGSERIAL    PRIMARY KEY,
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
    country_code     TEXT,
    CONSTRAINT uq_air_quality_loc_dt UNIQUE (location_id, datetime)
);

ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS country_code TEXT;

CREATE INDEX IF NOT EXISTS idx_air_quality_datetime  ON {TABLE_NAME} (datetime);
CREATE INDEX IF NOT EXISTS idx_air_quality_aqi_class ON {TABLE_NAME} (aqi_class);
CREATE INDEX IF NOT EXISTS idx_air_quality_location  ON {TABLE_NAME} (location_id);
CREATE INDEX IF NOT EXISTS idx_air_quality_country   ON {TABLE_NAME} (country_code);
"""

INSERT_SQL = f"""
INSERT INTO {TABLE_NAME} (
    datetime, location_id, name, provider_name, lat, lon,
    pm25, pm10, relativehumidity, temperature, um003,
    hour, day, month, year, dayofweek, aqi_class, aqi_label, country_code
) VALUES %s
ON CONFLICT (location_id, datetime) DO NOTHING
"""

COLUMNS_ORDER = [
    "datetime", "location_id", "name", "provider_name", "lat", "lon",
    "pm25", "pm10", "relativehumidity", "temperature", "um003",
    "hour", "day", "month", "year", "dayofweek", "aqi_class", "aqi_label",
    "country_code",
]


def prepare_df(parquet_path: Path) -> pd.DataFrame:
    """Carga el Parquet y lo adapta al schema de Supabase."""
    log.info("Leyendo %s ...", parquet_path)
    df = pd.read_parquet(parquet_path)
    log.info("Filas: %d | columnas: %s", len(df), list(df.columns))

    # station_name → name (matchea schema de Supabase)
    if "station_name" in df.columns:
        df = df.rename(columns={"station_name": "name"})

    # Asegura columnas que pueden no estar (NaN/None se mapean a NULL en Postgres)
    for col in ["pm10", "provider_name", "relativehumidity", "temperature", "um003"]:
        if col not in df.columns:
            df[col] = None

    if "pm25" not in df.columns:
        sys.exit("El Parquet no tiene la columna pm25 — no se puede subir.")

    # Clip valores fuera de rango físicamente razonable.
    CLIPS = {
        "pm25":             (0,    9000),
        "pm10":             (0,    9000),
        "pm1":              (0,    9000),
        "relativehumidity": (0,    100),
        "temperature":      (-80,  80),
        "um003":            (0,    1e9),
    }
    for col, (lo, hi) in CLIPS.items():
        if col in df.columns:
            n_out = ((df[col] < lo) | (df[col] > hi)).sum()
            if n_out > 0:
                log.info("  %s: %d valores fuera de [%s, %s] → clipped", col, n_out, lo, hi)
                df[col] = df[col].clip(lower=lo, upper=hi)

    df["datetime"] = (
        pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    )

    return df[COLUMNS_ORDER]


def open_connection() -> psycopg2.extensions.connection:
    """Abre una conexión con TCP keepalive para evitar drops del pooler."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        sys.exit("ERROR: falta DATABASE_URL en .env (Supabase Connection string).")
    # keepalive_*: hace que el SO mande pings TCP cada 30s para que el pooler
    # no considere la conexión inactiva y la corte.
    conn = psycopg2.connect(
        db_url,
        connect_timeout=30,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )
    return conn


def insert_batch_with_retry(rows_batch: list[tuple], attempts: int = MAX_RETRIES) -> None:
    """Inserta un batch. Si la conexión se cae, reabre y reintenta con backoff."""
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            conn = open_connection()
            try:
                with conn.cursor() as cur:
                    execute_values(cur, INSERT_SQL, rows_batch, page_size=BATCH_SIZE)
                conn.commit()
            finally:
                conn.close()
            return  # éxito
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
            last_exc = exc
            wait = 2 ** i  # 1s, 2s, 4s, 8s, 16s
            log.warning("Conexión caída (intento %d/%d), reintento en %ds: %s",
                        i + 1, attempts, wait, exc)
            time.sleep(wait)
    raise RuntimeError(f"Falló tras {attempts} intentos: {last_exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument(
        "--truncate", action="store_true",
        help="Vacía la tabla antes de insertar.",
    )
    args = parser.parse_args()

    if not args.parquet.exists():
        sys.exit(f"No existe el Parquet: {args.parquet}")

    df = prepare_df(args.parquet)
    log.info("Después de alinear schema: %d filas listas para subir", len(df))

    # Reemplaza NaN/NaT por None para que Postgres reciba NULL
    df_clean = df.where(pd.notnull(df), None)
    rows = list(df_clean.itertuples(index=False, name=None))
    total = len(rows)

    # Paso inicial: CREATE TABLE IF NOT EXISTS + ALTER + (opcional) TRUNCATE
    # en una sola conexión efímera. Esto hace el script auto-suficiente para
    # un proyecto Supabase nuevo (sin tabla previa) o uno con schema viejo.
    log.info("Preparando schema (CREATE + ALTER + opcional TRUNCATE) ...")
    conn = open_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(ENSURE_SCHEMA_SQL)
            if args.truncate:
                log.warning("⚠️ TRUNCATE: borrando contenido de %s", TABLE_NAME)
                cur.execute(f"TRUNCATE TABLE {TABLE_NAME} RESTART IDENTITY")
        conn.commit()
    finally:
        conn.close()

    # Inserta en batches con reconexión automática por batch.
    # Esto significa: si la conexión cae a la mitad, perdemos solo
    # los batches recientes — los anteriores están commiteados.
    log.info(
        "Insertando en batches de %d (reconexión por batch). Total: %d filas",
        BATCH_SIZE, total,
    )
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        insert_batch_with_retry(batch)
        done = min(i + BATCH_SIZE, total)
        if (done // BATCH_SIZE) % COMMIT_EVERY == 0 or done == total:
            log.info("  %d / %d insertadas (%.1f%%)", done, total, 100 * done / total)

    log.info("✅ Upload completo. Verifica en Supabase → Table editor → air_quality")


if __name__ == "__main__":
    main()
