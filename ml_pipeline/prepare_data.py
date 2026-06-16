"""Prepara dataset multinacional para entrenar el RF (legacy: vía API + S3).

Para el pipeline actual recomendado, ver `prepare_data_athena.py` que usa
Athena + Glue (más rápido y escalable). Este script se conserva como
alternativa cuando no quieres setup de Athena.

Pipeline (corre en local; barato de iterar):
1. Lista estaciones con PM2.5 de los 10 países curados (vía API REST).
2. Para cada estación, lista sus archivos de mediciones en S3 público en una
   ventana temporal configurable.
3. Descarga en paralelo (ThreadPoolExecutor).
4. Combina todo, pivota long → wide (parameter pasa a ser columna).
5. Filtra a registros con pm25 + pm10 presentes (mínimo para entrenar).
6. Añade aqi_class (EPA breakpoints) + features temporales.
7. Imputa medianas en columnas meteorológicas.
8. Guarda como Parquet (local + opcional sube a S3).

Uso:
    python ml_pipeline/prepare_data.py                # corre con defaults
    python ml_pipeline/prepare_data.py --months 24    # 24 meses de histórico
    python ml_pipeline/prepare_data.py --upload-s3    # sube al bucket de datos

Variables de entorno:
    OPENAQ_API_KEY          (obligatoria)
    DATA_BUCKET             (opcional, default 'aqi-almaty-data')
    MAX_STATIONS_PER_COUNTRY (opcional, default 50)
    AWS_REGION              (opcional, default 'us-east-1')
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import pandas as pd
import requests
from botocore import UNSIGNED
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

# ─── Config ────────────────────────────────────────────────────────────────
COUNTRY_CODES = ["BD", "IN", "PK", "NP", "MN", "TH", "VN", "KZ", "ID", "MX"]
OPENAQ_API = "https://api.openaq.org/v3"
OPENAQ_BUCKET = "openaq-data-archive"
PM25_PARAM_ID = 2

DATA_BUCKET = os.getenv("DATA_BUCKET", "aqi-almaty-data")
MAX_STATIONS_PER_COUNTRY = int(os.getenv("MAX_STATIONS_PER_COUNTRY", "50"))
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
PARALLEL_WORKERS = 20

API_KEY = os.getenv("OPENAQ_API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# Breakpoints EPA para PM2.5 (5 clases AQI estándar de la EPA)
AQI_LABELS = {
    0: "Buena",
    1: "Moderada",
    2: "Dañina para grupos sensibles",
    3: "Dañina",
    4: "Muy dañina",
}

# Cliente S3 anónimo (OpenAQ es AWS Open Data público). Subimos el pool de
# conexiones para que match con los workers paralelos y no veamos warnings.
s3_public = boto3.client(
    "s3",
    config=Config(
        signature_version=UNSIGNED,
        max_pool_connections=PARALLEL_WORKERS * 2,
    ),
    region_name="us-east-1",
)

log = logging.getLogger("prepare-data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ─── 1. Estaciones por país ────────────────────────────────────────────────
def _get_with_retry(url: str, params: dict, attempts: int = 4) -> dict:
    """GET con retries + backoff exponencial. Tolera 5xx y 429 de OpenAQ."""
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code >= 500 or r.status_code == 429:
                raise requests.HTTPError(f"{r.status_code} en {url}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            wait = 2 ** i  # 1s, 2s, 4s, 8s
            log.warning("Reintento %d/%d en %ds: %s", i + 1, attempts, wait, e)
            time.sleep(wait)
    raise RuntimeError(f"Falló tras {attempts} intentos: {last_exc}")


def fetch_country_ids() -> dict[str, int]:
    data = _get_with_retry(f"{OPENAQ_API}/countries", {"limit": 300})
    return {c["code"]: c["id"] for c in data["results"]}


def list_pm25_locations(country_id: int) -> list[dict]:
    """Pagina si hace falta. OpenAQ recomienda limit<=100."""
    all_results: list[dict] = []
    page = 1
    page_size = 100
    while True:
        data = _get_with_retry(
            f"{OPENAQ_API}/locations",
            {
                "countries_id": country_id,
                "parameters_id": PM25_PARAM_ID,
                "limit": page_size,
                "page": page,
            },
        )
        results = data.get("results", [])
        all_results.extend(results)
        # corta cuando ya tienes suficiente o no hay más páginas
        if len(results) < page_size or len(all_results) >= MAX_STATIONS_PER_COUNTRY:
            break
        page += 1
    return all_results


# ─── 2. Archivos S3 por estación en ventana temporal ───────────────────────
def list_station_files(location_id: int, start: datetime, end: datetime) -> list[str]:
    """Lista las keys S3 de un location_id dentro del rango [start, end]."""
    keys: list[str] = []
    cursor = start.replace(day=1)
    while cursor <= end:
        prefix = (
            f"records/csv.gz/locationid={location_id}/"
            f"year={cursor.year}/month={cursor.month:02d}/"
        )
        try:
            resp = s3_public.list_objects_v2(
                Bucket=OPENAQ_BUCKET, Prefix=prefix, MaxKeys=1000
            )
            keys.extend(obj["Key"] for obj in resp.get("Contents", []))
        except Exception as e:
            log.debug("list_objects_v2 falló %s: %s", prefix, e)
        # avanzar 1 mes
        cursor = (cursor + timedelta(days=32)).replace(day=1)
    return keys


# ─── 3. Descarga de un archivo CSV.GZ ──────────────────────────────────────
def download_csv(key: str) -> pd.DataFrame | None:
    try:
        obj = s3_public.get_object(Bucket=OPENAQ_BUCKET, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()), compression="gzip")
    except Exception as e:
        log.debug("Falla descarga %s: %s", key, e)
        return None


# ─── Pipeline principal ────────────────────────────────────────────────────
def collect_raw(months_back: int) -> pd.DataFrame:
    """Devuelve DataFrame long con todas las mediciones descargadas."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30 * months_back)
    log.info("Ventana temporal: %s → %s", start.date(), end.date())

    log.info("[1/3] Resolviendo countries y listando estaciones ...")
    country_map = fetch_country_ids()
    stations_meta: list[dict] = []
    for code in COUNTRY_CODES:
        cid = country_map.get(code)
        if cid is None:
            log.warning("País %s no encontrado en OpenAQ", code)
            continue
        try:
            locs = list_pm25_locations(cid)[:MAX_STATIONS_PER_COUNTRY]
        except Exception as e:
            log.warning("  %s: falló listar estaciones, skip — %s", code, e)
            continue
        for loc in locs:
            loc["_country_code"] = code
        stations_meta.extend(locs)
        log.info("  %s: %d estaciones (capped a %d)", code, len(locs), MAX_STATIONS_PER_COUNTRY)
    log.info("Total estaciones: %d", len(stations_meta))

    log.info("[2/3] Listando archivos S3 por estación (en paralelo) ...")
    all_keys: list[str] = []
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
        futures = {
            ex.submit(list_station_files, s["id"], start, end): s["id"]
            for s in stations_meta
        }
        for fut in as_completed(futures):
            try:
                all_keys.extend(fut.result())
            except Exception as e:
                log.warning("list_station_files falló: %s", e)
    log.info("Total archivos a descargar: %d", len(all_keys))

    log.info("[3/3] Descargando CSVs en paralelo ...")
    dataframes: list[pd.DataFrame] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
        for fut in as_completed({ex.submit(download_csv, k): k for k in all_keys}):
            df = fut.result()
            if df is not None and not df.empty:
                dataframes.append(df)
            completed += 1
            if completed % 500 == 0:
                log.info("  %d / %d descargados", completed, len(all_keys))
    log.info("Archivos válidos: %d", len(dataframes))

    if not dataframes:
        raise RuntimeError("No se descargó nada. Revisa OPENAQ_API_KEY y la ventana.")

    return pd.concat(dataframes, ignore_index=True)


def pivot_to_wide(df_long: pd.DataFrame, stations_meta: list[dict]) -> pd.DataFrame:
    """Long → Wide y enriquece con metadata de la estación."""
    log.info("Pivoteando long → wide ...")
    df_long["datetime"] = pd.to_datetime(df_long["datetime"], utc=True, errors="coerce")
    df_long = df_long.dropna(subset=["datetime", "value"])
    df_long = df_long[df_long["value"] >= 0]

    df_wide = df_long.pivot_table(
        index=["location_id", "datetime", "lat", "lon"],
        columns="parameter",
        values="value",
        aggfunc="mean",
    ).reset_index()
    df_wide.columns.name = None

    # Adjuntar metadata (country_code) por location_id
    meta = pd.DataFrame(
        [{"location_id": s["id"], "country_code": s["_country_code"],
          "station_name": s["name"]} for s in stations_meta]
    )
    df_wide = df_wide.merge(meta, on="location_id", how="left")
    return df_wide


# Columnas que NUNCA dropeamos por NaN (claves para el modelo o identidad)
PROTECTED_COLS = {
    "location_id", "datetime", "lat", "lon", "pm25",
    "country_code", "station_name",
}
# Umbral: si una columna tiene >NAN_DROP_THRESHOLD de NaN, la dropeamos
NAN_DROP_THRESHOLD = 0.70


def add_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra, añade aqi_class + features temporales + imputa medianas."""
    log.info("Filtrando, imputando y derivando features ...")

    # Solo PM2.5 es estrictamente necesario (es de donde sale el target).
    # PM10 y meteorología se imputan si faltan (los relajamos para no perder países).
    df = df.dropna(subset=["pm25"])
    df = df[df["pm25"] > 0]

    # Dropea columnas con demasiados NaN (excepto las protegidas)
    nan_ratio = df.isna().mean()
    to_drop = [
        c for c in df.columns
        if c not in PROTECTED_COLS and nan_ratio.get(c, 0) > NAN_DROP_THRESHOLD
    ]
    if to_drop:
        log.info(
            "Dropeando columnas con >%d%% NaN: %s",
            int(NAN_DROP_THRESHOLD * 100), to_drop,
        )
        df = df.drop(columns=to_drop)

    # aqi_class derivado de PM2.5 (mismos breakpoints EPA del proyecto original)
    def _classify(pm: float) -> int:
        if pm <= 12.0:
            return 0
        if pm <= 35.4:
            return 1
        if pm <= 55.4:
            return 2
        if pm <= 150.4:
            return 3
        return 4

    df["aqi_class"] = df["pm25"].astype(float).apply(_classify).astype("int8")
    df["aqi_label"] = df["aqi_class"].map(AQI_LABELS)

    # Features temporales
    dt = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    df["hour"] = dt.dt.hour.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["month"] = dt.dt.month.astype("int8")
    df["year"] = dt.dt.year.astype("int16")
    df["dayofweek"] = dt.dt.dayofweek.astype("int8")

    # Imputar numéricas restantes con la mediana global
    numeric_cols = df.select_dtypes(include="float64").columns.tolist()
    # No imputar PM2.5 (ya filtramos) ni lat/lon (siempre presentes)
    impute_cols = [c for c in numeric_cols if c not in {"pm25", "lat", "lon"}]
    medians = {}
    for col in impute_cols:
        med = float(df[col].median()) if df[col].notna().any() else 0.0
        medians[col] = med
        df[col] = df[col].fillna(med)

    log.info("Medianas usadas para imputación:")
    for k, v in medians.items():
        log.info("  %s = %.3f", k, v)
    log.info("Shape final: %s", df.shape)
    log.info("Países en dataset final:\n%s", df["country_code"].value_counts())
    log.info("Distribución aqi_class:\n%s", df["aqi_class"].value_counts().sort_index())
    return df


def maybe_upload_to_s3(local_path: Path, bucket: str, key: str) -> None:
    """Sube a tu data bucket (usa credenciales del .env / IAM)."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    log.info("Subiendo %s → s3://%s/%s ...", local_path, bucket, key)
    s3.upload_file(str(local_path), bucket, key)
    log.info("✅ Subido.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=12, help="Meses de histórico (default 12)")
    parser.add_argument("--out", type=str, default="data/openaq_processed.parquet")
    parser.add_argument("--upload-s3", action="store_true", help="Sube al bucket DATA_BUCKET")
    args = parser.parse_args()

    if not API_KEY:
        log.error("OPENAQ_API_KEY no está definido. Añádelo a tu .env.")
        sys.exit(1)

    # Re-resolvemos stations_meta para tener el country_code asociado
    country_map = fetch_country_ids()
    stations_meta: list[dict] = []
    for code in COUNTRY_CODES:
        cid = country_map.get(code)
        if cid is None:
            continue
        try:
            locs = list_pm25_locations(cid)[:MAX_STATIONS_PER_COUNTRY]
        except Exception as e:
            log.warning("%s: skip (no se pudo listar): %s", code, e)
            continue
        for loc in locs:
            loc["_country_code"] = code
        stations_meta.extend(locs)

    df_long = collect_raw(args.months)
    df_wide = pivot_to_wide(df_long, stations_meta)
    df_final = add_features_and_target(df_wide)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(out_path, index=False)
    log.info("✅ Guardado: %s (%.2f MB)", out_path, out_path.stat().st_size / 1e6)

    if args.upload_s3:
        maybe_upload_to_s3(out_path, DATA_BUCKET, "processed/openaq_processed.parquet")


if __name__ == "__main__":
    main()
