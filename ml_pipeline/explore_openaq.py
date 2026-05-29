"""Exploración inicial de OpenAQ para nuestra lista curada de países.

Objetivo: validar el esquema y la cobertura ANTES de escribir el job de
SageMaker, evitando gastar créditos en entrenamientos que fallan por datos.

Comprueba 4 cosas:
1. La API REST responde y nos da los country_ids correctos.
2. Cada país de la lista curada tiene estaciones que miden PM2.5.
3. Podemos acceder al bucket público S3 de OpenAQ desde tu PC (sin AWS creds).
4. Imprimimos el schema de una estación y un fragmento de mediciones reales.

Requisitos:
    pip install requests pandas pyarrow boto3

Opcional pero recomendado (sube los límites de la API):
    1. Regístrate gratis en https://explore.openaq.org/account
    2. Genera una API key
    3. export OPENAQ_API_KEY=tu_key
"""
from __future__ import annotations

import gzip
import io
import os
import sys
from datetime import datetime

import boto3
import pandas as pd
import requests
from botocore import UNSIGNED
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()  # lee .env del proyecto si existe

# Lista curada de países (ISO 3166-1 alpha-2)
COUNTRY_CODES = ["BD", "IN", "PK", "NP", "MN", "TH", "VN", "KZ", "ID", "MX"]

OPENAQ_API = "https://api.openaq.org/v3"
S3_BUCKET = "openaq-data-archive"
PM25_PARAM_ID = 2  # En OpenAQ v3: pm25 = id 2

API_KEY = os.getenv("OPENAQ_API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# Cliente S3 anónimo (OpenAQ es un dataset público de AWS Open Data)
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")


def fetch_country_ids() -> dict[str, int]:
    """Devuelve {ISO_code: country_id} para toda la base de OpenAQ."""
    r = requests.get(
        f"{OPENAQ_API}/countries", params={"limit": 300}, headers=HEADERS, timeout=30
    )
    r.raise_for_status()
    return {c["code"]: c["id"] for c in r.json()["results"]}


def list_pm25_locations(country_id: int, limit: int = 1000) -> list[dict]:
    """Estaciones de un país que miden PM2.5."""
    r = requests.get(
        f"{OPENAQ_API}/locations",
        params={
            "countries_id": country_id,
            "parameters_id": PM25_PARAM_ID,
            "limit": limit,
        },
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["results"]


def s3_list_prefix(bucket: str, prefix: str, max_keys: int = 10) -> list[str]:
    """Lista archivos bajo un prefijo. Soporta '/' como pseudo-directorio."""
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys)
    return [obj["Key"] for obj in resp.get("Contents", [])]


def main() -> None:
    print("=" * 70)
    print("OpenAQ — exploración de datos para lista curada")
    print("=" * 70)
    print(f"Países      : {', '.join(COUNTRY_CODES)}")
    print(f"API key     : {'sí' if API_KEY else 'NO (rate limits más bajos)'}")
    print(f"Fecha       : {datetime.utcnow().isoformat()}Z")
    print()

    # ---------- 1. Mapear ISO → country_id ----------
    print("[1/4] Resolviendo country_ids vía API REST ...")
    try:
        country_map = fetch_country_ids()
    except Exception as e:
        print(f"  ❌ Error llamando a la API: {e}")
        sys.exit(1)

    resolved: dict[str, int] = {}
    for code in COUNTRY_CODES:
        cid = country_map.get(code)
        if cid is None:
            print(f"  {code}: ❌ no encontrado en OpenAQ")
        else:
            resolved[code] = cid
            print(f"  {code}: country_id={cid}")
    print()

    # ---------- 2. Contar estaciones con PM2.5 ----------
    print("[2/4] Contando estaciones con PM2.5 por país ...")
    all_locations: list[dict] = []
    for code, cid in resolved.items():
        try:
            locs = list_pm25_locations(cid)
        except Exception as e:
            print(f"  {code}: ❌ error — {e}")
            continue
        all_locations.extend(locs)
        print(f"  {code}: {len(locs)} estaciones")
    print(f"  TOTAL: {len(all_locations)} estaciones")
    print()

    if not all_locations:
        print("❌ No hay estaciones. Algo va mal con la API. Revisa OPENAQ_API_KEY.")
        sys.exit(1)

    # ---------- 3. Schema de una estación ----------
    print("[3/4] Schema de la primera estación (para diseñar la tabla `locations`):")
    sample = all_locations[0]
    for key, val in sample.items():
        preview = str(val)
        if len(preview) > 100:
            preview = preview[:97] + "..."
        print(f"  {key:25s} = {preview}")
    print()

    # ---------- 4. Acceso al S3 público ----------
    print("[4/4] Probando acceso anónimo al S3 público (openaq-data-archive) ...")
    try:
        # En OpenAQ los archivos están bajo records/csv.gz/ — listamos lo de más alto
        top = s3_list_prefix(S3_BUCKET, "records/csv.gz/", max_keys=5)
        print(f"  ✅ Conectado. Primeros 5 objetos bajo records/csv.gz/:")
        for path in top:
            print(f"     {path}")
    except Exception as e:
        print(f"  ❌ Error S3: {e}")
        sys.exit(1)
    print()

    # Intentar bajar UNA muestra de mediciones de la primera estación
    location_id = sample["id"]
    print(f"  Buscando archivos para location_id={location_id} ...")
    candidates = [
        f"records/csv.gz/locationid={location_id}/",
        f"measurements/csv.gz/locationid={location_id}/",
        f"records/parquet/locationid={location_id}/",
    ]
    found = None
    for prefix in candidates:
        try:
            keys = s3_list_prefix(S3_BUCKET, prefix, max_keys=3)
            if keys:
                found = (prefix, keys)
                break
        except Exception:
            continue

    if not found:
        print(f"  ⚠️ No se encontraron archivos en los prefijos probados:")
        for c in candidates:
            print(f"       s3://{S3_BUCKET}/{c}")
        print(f"     Lista manual:")
        print(f"       aws s3 ls s3://{S3_BUCKET}/records/csv.gz/ --no-sign-request")
        return

    prefix, sample_keys = found
    print(f"  ✅ Encontrado en: s3://{S3_BUCKET}/{prefix}")
    for k in sample_keys:
        print(f"     {k}")

    first_key = sample_keys[0]
    print(f"\n  Leyendo {first_key} para ver el schema de medidas ...")
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=first_key)
        body = obj["Body"].read()
        if first_key.endswith(".gz"):
            df = pd.read_csv(io.BytesIO(body), compression="gzip", nrows=5)
        elif first_key.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(body)).head(5)
        else:
            df = pd.read_csv(io.BytesIO(body), nrows=5)
        print(f"  Columnas: {list(df.columns)}")
        print(f"  Primeras filas:")
        print(df.to_string(index=False, max_cols=8))
    except Exception as e:
        print(f"  ⚠️ No se pudo leer el archivo: {e}")

    print()
    print("=" * 70)
    print("✅ Exploración terminada. Comparte la salida y diseñamos `train.py`.")
    print("=" * 70)


if __name__ == "__main__":
    main()
