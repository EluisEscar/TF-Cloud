"""Configura Athena + Glue para consultar el S3 público de OpenAQ.

Ejecuta SOLO la primera vez (es idempotente: si la DB o la tabla ya
existen, no las recrea).

Crea:
1. Database Glue ``openaq_aqi``.
2. External table ``records`` apuntando a s3://openaq-data-archive/records/csv.gz/
   con **partition projection** (no necesitamos MSCK REPAIR — Athena
   resuelve los paths al vuelo según el WHERE).

Después de correr este script puedes ir a la consola de Athena y validar
con:

    SELECT location_id, datetime, parameter, value
    FROM openaq_aqi.records
    WHERE locationid = 22 AND year = 2025 AND month = 3
    LIMIT 10;

Uso:
    python ml_pipeline/athena_setup.py

Variables de entorno (desde .env):
    AWS_REGION                 default 'us-east-1'
    ATHENA_RESULTS_BUCKET      default 'aqi-athena-results-ee'
    GLUE_DATABASE              default 'openaq_aqi'
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import boto3
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("athena-setup")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
RESULTS_BUCKET = os.getenv("ATHENA_RESULTS_BUCKET", "aqi-athena-results-ee")
GLUE_DATABASE = os.getenv("GLUE_DATABASE", "openaq_aqi")
TABLE_NAME = "records"

# Esquema OpenAQ. Las columnas particionadas (locationid, year, month) van
# en PARTITIONED BY; el resto en la definición principal.
# Usamos OpenCSVSerde — todas las columnas llegan como STRING y casteamos
# en las queries finales.
CREATE_TABLE_SQL = f"""
CREATE EXTERNAL TABLE IF NOT EXISTS {GLUE_DATABASE}.{TABLE_NAME} (
  location_id INT,
  sensors_id  INT,
  location    STRING,
  datetime    STRING,
  lat         DOUBLE,
  lon         DOUBLE,
  parameter   STRING,
  units       STRING,
  value       DOUBLE
)
PARTITIONED BY (
  locationid INT,
  year       INT,
  month      INT
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar'     = '"',
  'escapeChar'    = '\\\\'
)
LOCATION 's3://openaq-data-archive/records/csv.gz/'
TBLPROPERTIES (
  'skip.header.line.count' = '1',
  'classification'         = 'csv',
  'projection.enabled'     = 'true',

  'projection.locationid.type'     = 'integer',
  'projection.locationid.range'    = '1,10000000',

  'projection.year.type'  = 'integer',
  'projection.year.range' = '2013,2030',

  'projection.month.type'   = 'integer',
  'projection.month.range'  = '1,12',
  'projection.month.digits' = '2',

  'storage.location.template' =
    's3://openaq-data-archive/records/csv.gz/locationid=${{locationid}}/year=${{year}}/month=${{month}}/'
)
""".strip()


def run_athena(client: Any, sql: str, results_uri: str, label: str) -> str:
    """Ejecuta una sentencia DDL en Athena, espera, devuelve QueryExecutionId."""
    log.info("Ejecutando: %s", label)
    res = client.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": results_uri},
    )
    qid = res["QueryExecutionId"]
    while True:
        r = client.get_query_execution(QueryExecutionId=qid)
        state = r["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(1.5)
    if state != "SUCCEEDED":
        reason = r["QueryExecution"]["Status"].get("StateChangeReason", "")
        raise RuntimeError(f"{label} terminó en {state}: {reason}")
    log.info("  ✅ %s", label)
    return qid


def main() -> None:
    log.info("Region        = %s", AWS_REGION)
    log.info("Results bucket = s3://%s", RESULTS_BUCKET)
    log.info("Glue database = %s", GLUE_DATABASE)
    log.info("Table         = %s.%s", GLUE_DATABASE, TABLE_NAME)

    glue = boto3.client("glue", region_name=AWS_REGION)
    athena = boto3.client("athena", region_name=AWS_REGION)
    results_uri = f"s3://{RESULTS_BUCKET}/queries/"

    # 1. Crear database si no existe
    try:
        glue.get_database(Name=GLUE_DATABASE)
        log.info("Database %s ya existe — skip", GLUE_DATABASE)
    except glue.exceptions.EntityNotFoundException:
        log.info("Creando database %s ...", GLUE_DATABASE)
        glue.create_database(DatabaseInput={
            "Name": GLUE_DATABASE,
            "Description": "Acceso a OpenAQ Open Data via Athena (proyecto AQI)",
        })
        log.info("  ✅ Database creada")

    # 2. Crear tabla externa (idempotente por IF NOT EXISTS)
    run_athena(athena, CREATE_TABLE_SQL, results_uri, "CREATE EXTERNAL TABLE")

    # 3. Smoke test: query rápida sobre una estación conocida (SPARTAN Dhaka)
    smoke_sql = f"""
        SELECT COUNT(*) AS n
        FROM {GLUE_DATABASE}.{TABLE_NAME}
        WHERE locationid = 22 AND year = 2025 AND month = 1
    """.strip()
    qid = run_athena(athena, smoke_sql, results_uri, "Smoke test (count)")

    # Lee el resultado del smoke test
    res = athena.get_query_results(QueryExecutionId=qid)
    rows = res["ResultSet"]["Rows"]
    if len(rows) > 1:
        count_value = rows[1]["Data"][0].get("VarCharValue", "0")
        log.info("Smoke test devuelve %s filas para location_id=22, 2025-01", count_value)

    log.info("✅ Setup completo. Ya puedes correr prepare_data_athena.py")


if __name__ == "__main__":
    main()
