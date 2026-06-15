# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline AQI multinacional — autónomo (Athena → entrenamiento → S3)
# MAGIC
# MAGIC Notebook **autónomo** que ejecuta el pipeline ETL + ML de extremo a extremo:
# MAGIC
# MAGIC 1. Lista estaciones con PM2.5 por país vía OpenAQ REST API.
# MAGIC 2. Ejecuta una query Athena contra el bucket público de OpenAQ
# MAGIC    (`s3://openaq-data-archive/`) con una ventana móvil configurable.
# MAGIC 3. Descarga el resultado del bucket de Athena.
# MAGIC 4. Pivota long → wide, deriva `aqi_class`, label-encodea `country_code`.
# MAGIC 5. Entrena `RandomForestClassifier` con `class_weight=balanced`.
# MAGIC 6. Gate de calidad (`accuracy_test >= MIN_ACCURACY`) → sube a S3.
# MAGIC
# MAGIC **No depende de archivos locales ni del Volume** — todo se obtiene en runtime
# MAGIC desde AWS + OpenAQ. Listo para programarse como Databricks Job semanal.
# MAGIC
# MAGIC ## Requisitos previos (una vez)
# MAGIC
# MAGIC Secrets en scope `aqi`:
# MAGIC - `openaq_api_key`         → API key gratis de https://explore.openaq.org
# MAGIC - `aws_access_key_id`      → IAM user con permisos Athena + S3 read/write
# MAGIC - `aws_secret_access_key`  →
# MAGIC
# MAGIC CLI:
# MAGIC ```
# MAGIC databricks secrets put-secret aqi openaq_api_key
# MAGIC databricks secrets put-secret aqi aws_access_key_id
# MAGIC databricks secrets put-secret aqi aws_secret_access_key
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q "scikit-learn==1.2.2" "boto3>=1.35" "pandas>=2.2,<3" "pyarrow>=14" "requests>=2.31" joblib
# MAGIC %restart_python

# COMMAND ----------

# Widgets de configuración (también los lee el Job programado)
dbutils.widgets.text("months_back", "12", "Meses de histórico a consultar")
dbutils.widgets.text("max_stations_per_country", "50", "Máx estaciones por país")
dbutils.widgets.text("model_bucket", "aqi-almaty-models-ee", "S3 bucket destino del modelo")
dbutils.widgets.text("athena_results_bucket", "aqi-athena-results-ee", "S3 bucket de resultados Athena")
dbutils.widgets.text("glue_database", "openaq_aqi", "Glue database")
dbutils.widgets.text("aws_region", "us-east-1", "AWS region")
dbutils.widgets.text("min_accuracy", "0.70", "Gate: accuracy mínima para publicar")

MONTHS_BACK = int(dbutils.widgets.get("months_back"))
MAX_STATIONS_PER_COUNTRY = int(dbutils.widgets.get("max_stations_per_country"))
MODEL_BUCKET = dbutils.widgets.get("model_bucket")
ATHENA_RESULTS_BUCKET = dbutils.widgets.get("athena_results_bucket")
GLUE_DATABASE = dbutils.widgets.get("glue_database")
AWS_REGION = dbutils.widgets.get("aws_region")
MIN_ACCURACY = float(dbutils.widgets.get("min_accuracy"))

print(f"Ventana temporal       : últimos {MONTHS_BACK} meses")
print(f"Máx estaciones por país: {MAX_STATIONS_PER_COUNTRY}")
print(f"Model bucket           : s3://{MODEL_BUCKET}")
print(f"Athena results bucket  : s3://{ATHENA_RESULTS_BUCKET}")
print(f"Glue database          : {GLUE_DATABASE}")
print(f"Gate accuracy mínima   : {MIN_ACCURACY}")

# COMMAND ----------

# MAGIC %md ## 1. Setup de clientes AWS y constantes

# COMMAND ----------

import boto3
import requests

# Credenciales AWS desde Databricks secrets
aws_key = dbutils.secrets.get(scope="aqi", key="aws_access_key_id")
aws_secret = dbutils.secrets.get(scope="aqi", key="aws_secret_access_key")
openaq_api_key = dbutils.secrets.get(scope="aqi", key="openaq_api_key")

# Clientes boto3
athena_client = boto3.client(
    "athena",
    aws_access_key_id=aws_key,
    aws_secret_access_key=aws_secret,
    region_name=AWS_REGION,
)
s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_key,
    aws_secret_access_key=aws_secret,
    region_name=AWS_REGION,
)

# Lista curada de países con alta PM2.5 + buena cobertura OpenAQ
COUNTRY_CODES = ["BD", "IN", "PK", "NP", "MN", "TH", "VN", "KZ", "ID", "MX"]
OPENAQ_API = "https://api.openaq.org/v3"
PM25_PARAM_ID = 2
OPENAQ_HEADERS = {"X-API-Key": openaq_api_key}
RELEVANT_PARAMETERS = ["pm25", "pm10", "pm1", "no2", "o3", "so2", "co",
                       "temperature", "relativehumidity", "um003",
                       "wind_speed", "wind_direction"]

# Etiquetas EPA
AQI_LABELS = {
    0: "Buena",
    1: "Moderada",
    2: "Dañina para grupos sensibles",
    3: "Dañina",
    4: "Muy dañina",
}

print("✓ Clientes AWS y secrets configurados")

# COMMAND ----------

# MAGIC %md ## 2. Listar estaciones por país (OpenAQ REST API)

# COMMAND ----------

import time

def api_get_with_retry(url, params, attempts=4):
    last_exc = None
    for i in range(attempts):
        try:
            r = requests.get(url, params=params, headers=OPENAQ_HEADERS, timeout=30)
            if r.status_code >= 500 or r.status_code == 429:
                raise requests.HTTPError(f"{r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            time.sleep(2 ** i)
    raise RuntimeError(f"Falló tras {attempts} intentos: {last_exc}")


def fetch_country_ids():
    data = api_get_with_retry(f"{OPENAQ_API}/countries", {"limit": 300})
    return {c["code"]: c["id"] for c in data["results"]}


def list_pm25_locations(country_id):
    all_results = []
    page = 1
    while True:
        data = api_get_with_retry(
            f"{OPENAQ_API}/locations",
            {"countries_id": country_id, "parameters_id": PM25_PARAM_ID,
             "limit": 100, "page": page},
        )
        results = data.get("results", [])
        all_results.extend(results)
        if len(results) < 100 or len(all_results) >= MAX_STATIONS_PER_COUNTRY:
            break
        page += 1
    return all_results[:MAX_STATIONS_PER_COUNTRY]


country_map = fetch_country_ids()
stations_meta = []
for code in COUNTRY_CODES:
    cid = country_map.get(code)
    if cid is None:
        continue
    try:
        locs = list_pm25_locations(cid)
    except Exception as e:
        print(f"  {code}: skip ({e})")
        continue
    for loc in locs:
        loc["_country_code"] = code
    stations_meta.extend(locs)
    print(f"  {code}: {len(locs)} estaciones")

location_ids = [s["id"] for s in stations_meta]
print(f"\nTotal location_ids para query: {len(location_ids)}")

# COMMAND ----------

# MAGIC %md ## 3. Ejecutar query en Athena

# COMMAND ----------

from datetime import datetime, timedelta, timezone

def build_athena_query(location_ids, start_dt, end_dt):
    ids_csv = ",".join(str(i) for i in sorted(set(location_ids)))
    months = []
    cursor = start_dt.replace(day=1)
    while cursor <= end_dt:
        months.append(f"(year = {cursor.year} AND month = {cursor.month})")
        cursor = (cursor + timedelta(days=32)).replace(day=1)
    months_clause = " OR ".join(months)
    parameters_csv = ",".join(f"'{p}'" for p in RELEVANT_PARAMETERS)
    return f"""
SELECT location_id, datetime, lat, lon, parameter, value, locationid
FROM {GLUE_DATABASE}.records
WHERE locationid IN ({ids_csv})
  AND ({months_clause})
  AND parameter IN ({parameters_csv})
  AND value IS NOT NULL
""".strip()


def run_athena_query(sql):
    results_uri = f"s3://{ATHENA_RESULTS_BUCKET}/queries/"
    res = athena_client.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": results_uri},
    )
    qid = res["QueryExecutionId"]
    print(f"QueryExecutionId: {qid}")

    while True:
        r = athena_client.get_query_execution(QueryExecutionId=qid)
        state = r["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(3)

    if state != "SUCCEEDED":
        reason = r["QueryExecution"]["Status"].get("StateChangeReason", "")
        raise RuntimeError(f"Query terminó en {state}: {reason}")

    stats = r["QueryExecution"]["Statistics"]
    scanned_mb = stats.get("DataScannedInBytes", 0) / 1e6
    runtime_s = stats.get("TotalExecutionTimeInMillis", 0) / 1000
    print(f"Query OK · escaneó {scanned_mb:.1f} MB · {runtime_s:.1f}s")

    output = r["QueryExecution"]["ResultConfiguration"]["OutputLocation"]
    bucket, key = output.replace("s3://", "").split("/", 1)
    return bucket, key


end_dt = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(days=30 * MONTHS_BACK)
print(f"Ventana: {start_dt.date()} → {end_dt.date()}")

sql = build_athena_query(location_ids, start_dt, end_dt)
result_bucket, result_key = run_athena_query(sql)
print(f"Resultado en: s3://{result_bucket}/{result_key}")

# COMMAND ----------

# MAGIC %md ## 4. Descargar resultado y procesar (long → wide + features)

# COMMAND ----------

import io
import pandas as pd

# Descarga el CSV de resultados de Athena
print(f"Descargando s3://{result_bucket}/{result_key} ...")
obj = s3_client.get_object(Bucket=result_bucket, Key=result_key)
df_long = pd.read_csv(io.BytesIO(obj["Body"].read()))
print(f"Filas crudas: {len(df_long):,}")

if df_long.empty:
    raise RuntimeError("Athena devolvió 0 filas. Revisa la query.")

# Pivot long → wide
print("Pivoteando long → wide ...")
df_long["datetime"] = pd.to_datetime(df_long["datetime"], utc=True, errors="coerce")
df_long = df_long.dropna(subset=["datetime", "value"])
df_long = df_long[df_long["value"] >= 0]

df = df_long.pivot_table(
    index=["location_id", "datetime", "lat", "lon"],
    columns="parameter",
    values="value",
    aggfunc="mean",
).reset_index()
df.columns.name = None

# Merge con metadata de estaciones (country_code, station_name)
meta = pd.DataFrame([
    {"location_id": s["id"], "country_code": s["_country_code"],
     "station_name": s["name"]}
    for s in stations_meta
])
df = df.merge(meta, on="location_id", how="left")
print(f"Después de pivot: {df.shape}")

# Filtra: necesitamos pm25 (target) y al menos un predictor
df = df.dropna(subset=["pm25"])
df = df[df["pm25"] > 0]

# Drop columnas con >70% NaN (excepto protegidas)
PROTECTED_COLS = {"location_id", "datetime", "lat", "lon", "pm25",
                  "country_code", "station_name"}
NAN_THRESHOLD = 0.70
nan_ratio = df.isna().mean()
to_drop = [c for c in df.columns
           if c not in PROTECTED_COLS and nan_ratio.get(c, 0) > NAN_THRESHOLD]
if to_drop:
    print(f"Dropeando columnas >70% NaN: {to_drop}")
    df = df.drop(columns=to_drop)

# Derivar aqi_class desde pm25
def _classify(pm):
    if pm <= 12.0: return 0
    if pm <= 35.4: return 1
    if pm <= 55.4: return 2
    if pm <= 150.4: return 3
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

# Imputar numéricas con mediana global
numeric_cols = df.select_dtypes(include="float64").columns.tolist()
NO_IMPUTE = {"lat", "lon", "hour", "day", "month", "year", "dayofweek", "aqi_class"}
impute_cols = [c for c in numeric_cols if c not in NO_IMPUTE]
medians = {}
for col in impute_cols:
    med = float(df[col].median()) if df[col].notna().any() else 0.0
    medians[col] = med
    df[col] = df[col].fillna(med)

print(f"\nShape final: {df.shape}")
print(f"\nPaíses:\n{df['country_code'].value_counts()}")
print(f"\nDistribución aqi_class:\n{df['aqi_class'].value_counts().sort_index()}")

# COMMAND ----------

# MAGIC %md ## 5. Preparar X / y (con country_id label-encoded)

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

CANDIDATE_NUMERIC = [
    "pm1", "pm10", "relativehumidity", "temperature", "um003",
    "co", "no2", "o3", "so2", "wind_speed", "wind_direction",
    "lat", "lon",
]
TEMPORAL_COLS = ["hour", "day", "month", "year", "dayofweek"]
TARGET = "aqi_class"

# Label-encode country_code → country_id
encoder = LabelEncoder()
df["country_id"] = encoder.fit_transform(df["country_code"].astype(str))
country_mapping = {
    code: int(idx)
    for code, idx in zip(encoder.classes_, encoder.transform(encoder.classes_))
}
print(f"country_encoder: {country_mapping}")

numeric_cols_final = [c for c in CANDIDATE_NUMERIC if c in df.columns]
feature_cols = numeric_cols_final + TEMPORAL_COLS + ["country_id"]
print(f"Features ({len(feature_cols)}): {feature_cols}")

X = df[feature_cols].astype(float)
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# COMMAND ----------

# MAGIC %md ## 6. Entrenar
# MAGIC
# MAGIC > Sin MLflow: Databricks Free Edition (serverless) no expone
# MAGIC > `spark.mlflow.modelRegistryUri`. El modelo se publica directo a S3.

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_leaf": 50,
    "class_weight": "balanced",
    "random_state": 42,
}

print("Entrenando RandomForestClassifier ...")
rf = RandomForestClassifier(n_jobs=-1, **PARAMS)
rf.fit(X_train, y_train)
print("Modelo entrenado ✓")

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")

print(f"\naccuracy_test = {acc:.4f}")
print(f"f1_macro      = {f1_macro:.4f}\n")
print(classification_report(
    y_test, y_pred,
    target_names=[AQI_LABELS[i] for i in sorted(AQI_LABELS)],
    digits=4, zero_division=0,
))

# COMMAND ----------

# MAGIC %md ## 7. Serializar + publicar a S3 (con gate de calidad)

# COMMAND ----------

import json
import joblib
import os

medians_final = {
    c: float(df[c].median())
    for c in numeric_cols_final if c not in NO_IMPUTE and df[c].notna().any()
}

metadata = {
    "feature_names": feature_cols,
    "target": TARGET,
    "class_labels": {str(k): v for k, v in AQI_LABELS.items()},
    "medians": medians_final,
    "country_encoder": country_mapping,
    "model_type": "RandomForestClassifier",
    "params": PARAMS,
    "metrics": {"accuracy_test": float(acc), "f1_macro": float(f1_macro)},
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "training_source": "databricks-autonomous",
    "data_source": "openaq-athena",
    "countries": sorted(country_mapping.keys()),
    "months_back": MONTHS_BACK,
    "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
}

local_model = "/tmp/rf_aqi.pkl"
local_meta = "/tmp/features.json"
joblib.dump(rf, local_model)
with open(local_meta, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Tamaño .pkl: {round(os.path.getsize(local_model) / 1e6, 2)} MB")

# Gate de calidad
if acc < MIN_ACCURACY:
    raise ValueError(
        f"❌ accuracy_test={acc:.4f} < umbral {MIN_ACCURACY}. NO se publica."
    )

s3_client.upload_file(local_model, MODEL_BUCKET, "rf_aqi.pkl")
s3_client.upload_file(local_meta, MODEL_BUCKET, "features.json")
print(f"\n✅ Modelo publicado en s3://{MODEL_BUCKET}/")
print(f"   accuracy_test = {acc:.4f}")
print(f"   training_timestamp = {metadata['training_timestamp_utc']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Listo
# MAGIC
# MAGIC Este notebook es **completamente autónomo**:
# MAGIC - No requiere subir archivos a un Volume.
# MAGIC - No depende del PC del usuario.
# MAGIC - Configura solo los AWS clients vía secrets y ejecuta de inicio a fin.
# MAGIC
# MAGIC Para automatizarlo: **Workflows → Create job** apuntando a este notebook,
# MAGIC con schedule `0 0 * * 0` (domingos 00:00 UTC) o el cron que prefieras.
# MAGIC
# MAGIC La EC2 descargará el `.pkl` nuevo al reiniciarse. Para refrescar automá-
# MAGIC ticamente después del job, opciones:
# MAGIC 1. **Manual:** `docker restart aqi-api` después de cada reentreno.
# MAGIC 2. **Automático:** Lambda con trigger `s3:ObjectCreated` que llame a
# MAGIC    Systems Manager Run Command sobre la EC2 con el `docker restart`.
