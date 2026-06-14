"""Versión Athena de prepare_data.py.

Reemplaza el patrón "API + descarga paralela de miles de CSV.gz" por una
sola query SQL contra el bucket público de OpenAQ vía Athena. Mucho más
elegante y permite ventanas temporales largas (12-24 meses) sin sufrir.

Flujo:
1. Listar estaciones por país vía la API REST de OpenAQ (igual que antes).
2. Construir una query Athena con WHERE locationid IN (...) y un rango
   year/month. Athena resuelve solo las particiones necesarias gracias
   a la partition projection definida en athena_setup.py.
3. Ejecutar y esperar resultado.
4. Descargar el CSV de resultados desde el bucket de Athena.
5. Procesar igual que antes: pivot long→wide, AQI class, imputación.
6. Guardar Parquet local (+ opcional upload al data bucket).

Uso:
    python ml_pipeline/prepare_data_athena.py                       # 12 meses default
    python ml_pipeline/prepare_data_athena.py --months 24
    python ml_pipeline/prepare_data_athena.py --upload-s3

Variables de entorno (.env):
    OPENAQ_API_KEY            obligatoria
    AWS_REGION                default us-east-1
    ATHENA_RESULTS_BUCKET     default aqi-athena-results-ee
    GLUE_DATABASE             default openaq_aqi
    DATA_BUCKET               default aqi-almaty-data-ee
    MAX_STATIONS_PER_COUNTRY  default 50
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ─── Config ────────────────────────────────────────────────────────────────
COUNTRY_CODES = ["BD", "IN", "PK", "NP", "MN", "TH", "VN", "KZ", "ID", "MX"]
OPENAQ_API = "https://api.openaq.org/v3"
PM25_PARAM_ID = 2

# Parámetros que conservamos del raw — usaremos los que estén con valores
# significativos durante el procesamiento.
RELEVANT_PARAMETERS = ["pm25", "pm10", "pm1", "no2", "o3", "so2", "co",
                       "temperature", "relativehumidity", "um003",
                       "wind_speed", "wind_direction"]

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
RESULTS_BUCKET = os.getenv("ATHENA_RESULTS_BUCKET", "aqi-athena-results-ee")
GLUE_DATABASE = os.getenv("GLUE_DATABASE", "openaq_aqi")
DATA_BUCKET = os.getenv("DATA_BUCKET", "aqi-almaty-data-ee")
MAX_STATIONS_PER_COUNTRY = int(os.getenv("MAX_STATIONS_PER_COUNTRY", "50"))

API_KEY = os.getenv("OPENAQ_API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

AQI_LABELS = {
    0: "Buena",
    1: "Moderada",
    2: "Dañina para grupos sensibles",
    3: "Dañina",
    4: "Muy dañina",
}

PROTECTED_COLS = {"location_id", "datetime", "lat", "lon", "pm25",
                  "country_code", "station_name"}
NAN_DROP_THRESHOLD = 0.70

log = logging.getLogger("athena-prep")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

athena = boto3.client("athena", region_name=AWS_REGION)
s3 = boto3.client("s3", region_name=AWS_REGION)


# ─── 1. Estaciones por país (vía API REST) ─────────────────────────────────
def _api_get_with_retry(url: str, params: dict, attempts: int = 4) -> dict:
    last_exc = None
    for i in range(attempts):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code >= 500 or r.status_code == 429:
                raise requests.HTTPError(f"{r.status_code} en {url}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            wait = 2 ** i
            log.warning("Reintento %d/%d en %ds: %s", i + 1, attempts, wait, e)
            time.sleep(wait)
    raise RuntimeError(f"Falló tras {attempts} intentos: {last_exc}")


def fetch_country_ids() -> dict[str, int]:
    data = _api_get_with_retry(f"{OPENAQ_API}/countries", {"limit": 300})
    return {c["code"]: c["id"] for c in data["results"]}


def list_pm25_locations(country_id: int) -> list[dict]:
    all_results = []
    page = 1
    while True:
        data = _api_get_with_retry(
            f"{OPENAQ_API}/locations",
            {"countries_id": country_id, "parameters_id": PM25_PARAM_ID,
             "limit": 100, "page": page},
        )
        results = data.get("results", [])
        all_results.extend(results)
        if len(results) < 100 or len(all_results) >= MAX_STATIONS_PER_COUNTRY:
            break
        page += 1
    return all_results


# ─── 2. Athena query ───────────────────────────────────────────────────────
def build_athena_query(
    location_ids: list[int], start: datetime, end: datetime,
) -> str:
    """Construye la query SQL. Filtra por location_ids + ventana de meses."""
    ids_csv = ",".join(str(i) for i in sorted(set(location_ids)))
    # Rango de meses: la query maneja años parciales generando condiciones
    # explícitas (más fácil que CAST de partition values).
    months = []
    cursor = start.replace(day=1)
    while cursor <= end:
        months.append(f"(year = {cursor.year} AND month = {cursor.month})")
        cursor = (cursor + timedelta(days=32)).replace(day=1)
    months_clause = " OR ".join(months)

    parameters_csv = ",".join(f"'{p}'" for p in RELEVANT_PARAMETERS)

    return f"""
SELECT location_id,
       datetime,
       lat,
       lon,
       parameter,
       value,
       locationid
FROM {GLUE_DATABASE}.records
WHERE locationid IN ({ids_csv})
  AND ({months_clause})
  AND parameter IN ({parameters_csv})
  AND value IS NOT NULL
""".strip()


def run_query_and_get_result_path(sql: str) -> tuple[str, str]:
    """Ejecuta query, espera, devuelve (bucket, key) del CSV de resultado."""
    results_uri = f"s3://{RESULTS_BUCKET}/queries/"
    log.info("Lanzando query Athena ...")
    log.debug(sql)
    res = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": results_uri},
    )
    qid = res["QueryExecutionId"]
    log.info("QueryExecutionId: %s", qid)

    # Polling
    while True:
        r = athena.get_query_execution(QueryExecutionId=qid)
        state = r["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(3)

    if state != "SUCCEEDED":
        reason = r["QueryExecution"]["Status"].get("StateChangeReason", "")
        raise RuntimeError(f"Query terminó en {state}: {reason}")

    stats = r["QueryExecution"]["Statistics"]
    scanned_mb = stats.get("DataScannedInBytes", 0) / 1e6
    runtime_ms = stats.get("TotalExecutionTimeInMillis", 0)
    log.info("Query OK · escaneó %.1f MB · %.1fs", scanned_mb, runtime_ms / 1000)

    # La ruta del CSV es: s3://results-bucket/queries/<qid>.csv
    output = r["QueryExecution"]["ResultConfiguration"]["OutputLocation"]
    bucket, key = output.replace("s3://", "").split("/", 1)
    return bucket, key


def download_query_csv(bucket: str, key: str) -> pd.DataFrame:
    log.info("Descargando resultado s3://%s/%s ...", bucket, key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    log.info("Filas crudas descargadas: %d", len(df))
    return df


# ─── 3. Procesamiento (igual que prepare_data.py) ──────────────────────────
def pivot_to_wide(df_long: pd.DataFrame, stations_meta: list[dict]) -> pd.DataFrame:
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

    meta = pd.DataFrame([
        {"location_id": s["id"], "country_code": s["_country_code"],
         "station_name": s["name"]}
        for s in stations_meta
    ])
    df_wide = df_wide.merge(meta, on="location_id", how="left")
    return df_wide


def add_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Filtrando, derivando features ...")
    df = df.dropna(subset=["pm25"])
    df = df[df["pm25"] > 0]

    nan_ratio = df.isna().mean()
    to_drop = [c for c in df.columns
               if c not in PROTECTED_COLS and nan_ratio.get(c, 0) > NAN_DROP_THRESHOLD]
    if to_drop:
        log.info("Dropeando columnas con >%d%% NaN: %s",
                 int(NAN_DROP_THRESHOLD * 100), to_drop)
        df = df.drop(columns=to_drop)

    def _classify(pm: float) -> int:
        if pm <= 12.0: return 0
        if pm <= 35.4: return 1
        if pm <= 55.4: return 2
        if pm <= 150.4: return 3
        return 4

    df["aqi_class"] = df["pm25"].astype(float).apply(_classify).astype("int8")
    df["aqi_label"] = df["aqi_class"].map(AQI_LABELS)

    dt = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    df["hour"] = dt.dt.hour.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["month"] = dt.dt.month.astype("int8")
    df["year"] = dt.dt.year.astype("int16")
    df["dayofweek"] = dt.dt.dayofweek.astype("int8")

    numeric_cols = df.select_dtypes(include="float64").columns.tolist()
    impute_cols = [c for c in numeric_cols if c not in {"pm25", "lat", "lon"}]
    medians = {}
    for col in impute_cols:
        med = float(df[col].median()) if df[col].notna().any() else 0.0
        medians[col] = med
        df[col] = df[col].fillna(med)

    log.info("Medianas: %s", medians)
    log.info("Shape final: %s", df.shape)
    log.info("Países en dataset final:\n%s", df["country_code"].value_counts())
    log.info("Distribución aqi_class:\n%s", df["aqi_class"].value_counts().sort_index())
    return df


# ─── Pipeline principal ────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=12,
                        help="Meses de histórico (default 12)")
    parser.add_argument("--out", type=str,
                        default="data/openaq_processed.parquet")
    parser.add_argument("--upload-s3", action="store_true")
    args = parser.parse_args()

    if not API_KEY:
        log.error("OPENAQ_API_KEY no está definido. Añádelo a tu .env.")
        sys.exit(1)

    # 1. Listar estaciones
    log.info("[1/4] Listando estaciones de los 10 países curados ...")
    country_map = fetch_country_ids()
    stations_meta = []
    for code in COUNTRY_CODES:
        cid = country_map.get(code)
        if cid is None:
            continue
        try:
            locs = list_pm25_locations(cid)[:MAX_STATIONS_PER_COUNTRY]
        except Exception as e:
            log.warning("%s: skip (%s)", code, e)
            continue
        for loc in locs:
            loc["_country_code"] = code
        stations_meta.extend(locs)
        log.info("  %s: %d estaciones", code, len(locs))

    location_ids = [s["id"] for s in stations_meta]
    log.info("Total location_ids para query: %d", len(location_ids))

    # 2. Ejecutar query en Athena
    log.info("[2/4] Construyendo y lanzando query Athena ...")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30 * args.months)
    log.info("Ventana: %s → %s", start.date(), end.date())

    sql = build_athena_query(location_ids, start, end)
    bucket, key = run_query_and_get_result_path(sql)

    # 3. Descargar resultado
    log.info("[3/4] Descargando resultado ...")
    df_long = download_query_csv(bucket, key)
    if df_long.empty:
        log.error("Athena devolvió 0 filas. Revisa la query.")
        sys.exit(1)

    # 4. Procesar
    log.info("[4/4] Procesando ...")
    df_wide = pivot_to_wide(df_long, stations_meta)
    df_final = add_features_and_target(df_wide)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(out_path, index=False)
    log.info("✅ Guardado: %s (%.2f MB)", out_path, out_path.stat().st_size / 1e6)

    if args.upload_s3:
        log.info("Subiendo a s3://%s/processed/openaq_processed.parquet ...", DATA_BUCKET)
        s3.upload_file(str(out_path), DATA_BUCKET, "processed/openaq_processed.parquet")
        log.info("✅ Subido.")


if __name__ == "__main__":
    main()
