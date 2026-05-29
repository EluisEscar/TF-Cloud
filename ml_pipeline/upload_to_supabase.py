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
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = PROJECT_ROOT / "data" / "openaq_processed.parquet"
TABLE_NAME = "air_quality"
BATCH_SIZE = 5000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("upload")

# ALTER idempotente para añadir country_code a la tabla original
ENSURE_COLUMNS_SQL = f"""
ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS country_code TEXT;
CREATE INDEX IF NOT EXISTS idx_air_quality_country ON {TABLE_NAME} (country_code);
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

    # Si pm25 no está, no podemos cargar (es el target source) — pero por
    # construcción de prepare_data.py debería estar.
    if "pm25" not in df.columns:
        sys.exit("El Parquet no tiene la columna pm25 — no se puede subir.")

    # Clip valores fuera de rango físicamente razonable. Sensores OpenAQ a
    # veces reportan valores absurdos (10^7 µg/m³, temperaturas de 10000°C)
    # cuando están dañados, y Postgres rechaza por overflow de precisión
    # (NUMERIC(10,3) tope ~10^7). Limitamos a rangos físicos sensatos.
    CLIPS = {
        "pm25":             (0,    9000),   # rango EPA realista: 0-500, extremos hasta ~1500
        "pm10":             (0,    9000),
        "pm1":              (0,    9000),   # por si la columna está
        "relativehumidity": (0,    100),
        "temperature":      (-80,  80),     # min/max terrestre real
        "um003":            (0,    1e9),    # particle count, precision 12 lo aguanta
    }
    n_before = len(df)
    for col, (lo, hi) in CLIPS.items():
        if col in df.columns:
            n_out = ((df[col] < lo) | (df[col] > hi)).sum()
            if n_out > 0:
                log.info("  %s: %d valores fuera de [%s, %s] → clipped", col, n_out, lo, hi)
                df[col] = df[col].clip(lower=lo, upper=hi)

    # datetime a Python datetime (psycopg2 lo serializa bien)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)

    return df[COLUMNS_ORDER]


def get_connection() -> psycopg2.extensions.connection:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        sys.exit("ERROR: falta DATABASE_URL en .env (Supabase Connection string).")
    return psycopg2.connect(db_url)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument(
        "--truncate", action="store_true",
        help="Vacía la tabla antes de insertar (borra los datos viejos de Almaty).",
    )
    args = parser.parse_args()

    if not args.parquet.exists():
        sys.exit(f"No existe el Parquet: {args.parquet}\nCorrelo primero con prepare_data.py")

    df = prepare_df(args.parquet)
    log.info("Después de alinear schema: %d filas listas para subir", len(df))

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            log.info("ALTER TABLE (idempotente) para añadir country_code ...")
            cur.execute(ENSURE_COLUMNS_SQL)

            if args.truncate:
                log.warning("⚠️ TRUNCATE: borrando contenido de %s", TABLE_NAME)
                cur.execute(f"TRUNCATE TABLE {TABLE_NAME} RESTART IDENTITY")

            # Convierte a lista de tuplas para execute_values
            # Reemplaza NaN/NaT por None para que Postgres reciba NULL
            df_clean = df.where(pd.notnull(df), None)
            rows = list(df_clean.itertuples(index=False, name=None))

            total = len(rows)
            for i in range(0, total, BATCH_SIZE):
                batch = rows[i:i + BATCH_SIZE]
                execute_values(cur, INSERT_SQL, batch, page_size=BATCH_SIZE)
                log.info("  %d / %d insertadas", min(i + BATCH_SIZE, total), total)

        conn.commit()
        log.info("✅ Upload completo. Verifica en Supabase → Table editor → air_quality")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
