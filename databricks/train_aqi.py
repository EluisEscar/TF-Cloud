# Databricks notebook source
# MAGIC %md
# MAGIC # Reentrenamiento del modelo de calidad del aire (multinacional)
# MAGIC
# MAGIC Notebook de Databricks que reentrena el `RandomForestClassifier` sobre el
# MAGIC dataset OpenAQ multipaís (10 países, ~1.7M filas), registra el experimento
# MAGIC en **MLflow** y publica el modelo en **AWS S3** para que la API en EC2
# MAGIC lo descargue al arrancar.
# MAGIC
# MAGIC **Flujo:** Parquet en Volume → entrenamiento + MLflow → S3 → API en EC2.
# MAGIC
# MAGIC ## Requisitos previos (una sola vez)
# MAGIC
# MAGIC 1. **Volume creado** en Unity Catalog y `openaq_processed.parquet` subido.
# MAGIC 2. **Secrets** en Databricks (scope `aqi`):
# MAGIC    - `aqi/aws_access_key_id`     → del IAM user con permiso write al model bucket
# MAGIC    - `aqi/aws_secret_access_key`
# MAGIC
# MAGIC ## Cómo crear los secrets (Databricks CLI)
# MAGIC ```bash
# MAGIC databricks secrets create-scope aqi
# MAGIC databricks secrets put-secret aqi aws_access_key_id
# MAGIC databricks secrets put-secret aqi aws_secret_access_key
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -q "scikit-learn==1.2.2" "boto3>=1.35" "pandas>=2.2" "pyarrow>=14" joblib
# MAGIC %restart_python

# COMMAND ----------

# Widgets de parámetros
dbutils.widgets.text(
    "volume_path",
    "/Volumes/workspace/default/aqi/openaq_processed.parquet",
    "Ruta del Parquet en el Volume",
)
dbutils.widgets.text("model_bucket", "aqi-almaty-models-ee", "S3 bucket destino del modelo")
dbutils.widgets.text("aws_region", "us-east-1", "AWS region")
dbutils.widgets.text("min_accuracy", "0.65", "Umbral mínimo de accuracy para publicar")

VOLUME_PATH = dbutils.widgets.get("volume_path")
MODEL_BUCKET = dbutils.widgets.get("model_bucket")
AWS_REGION = dbutils.widgets.get("aws_region")
MIN_ACCURACY = float(dbutils.widgets.get("min_accuracy"))

print(f"Volume path   : {VOLUME_PATH}")
print(f"Model bucket  : s3://{MODEL_BUCKET}")
print(f"AWS region    : {AWS_REGION}")
print(f"Min accuracy  : {MIN_ACCURACY}")

# COMMAND ----------

# MAGIC %md ## 1. Cargar Parquet desde el Volume

# COMMAND ----------

import pandas as pd

df = pd.read_parquet(VOLUME_PATH)
print(f"Filas cargadas: {len(df):,}")
print(f"Columnas      : {list(df.columns)}")
df["aqi_class"].value_counts().sort_index()

# COMMAND ----------

# MAGIC %md ## 2. Preparar features (mismo schema que SageMaker)

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

# Candidatos numéricos (se incluyen solo los que existen en el Parquet)
CANDIDATE_NUMERIC = [
    "pm1", "pm10", "relativehumidity", "temperature", "um003",
    "co", "no2", "o3", "so2", "wind_speed", "wind_direction",
    "lat", "lon",
]
TEMPORAL_COLS = ["hour", "day", "month", "year", "dayofweek"]
TARGET = "aqi_class"

# Etiquetas EPA
AQI_LABELS = {
    0: "Buena",
    1: "Moderada",
    2: "Dañina para grupos sensibles",
    3: "Dañina",
    4: "Muy dañina",
}

# Label-encode country_code → country_id (categórica para el RF)
encoder = LabelEncoder()
df["country_id"] = encoder.fit_transform(df["country_code"].astype(str))
country_mapping = {
    code: int(idx)
    for code, idx in zip(encoder.classes_, encoder.transform(encoder.classes_))
}
print(f"country_encoder: {country_mapping}")

# Selecciona las features reales (solo las que vienen en el Parquet)
numeric_cols = [c for c in CANDIDATE_NUMERIC if c in df.columns]
feature_cols = numeric_cols + TEMPORAL_COLS + ["country_id"]
print(f"Features ({len(feature_cols)}): {feature_cols}")

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = df[feature_cols].astype(float)
y = df[TARGET].astype(int)
print(f"X: {X.shape} | y: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# COMMAND ----------

# MAGIC %md ## 3. Entrenar con tracking de MLflow

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_leaf": 50,
    "class_weight": "balanced",
    "random_state": 42,
}

# Nota: en Databricks Free Edition (serverless compute) `mlflow.set_experiment`
# con una ruta /Shared/ falla con CONFIG_NOT_AVAILABLE (el model registry no
# está disponible en el tier gratuito). Lo dejamos comentado — MLflow usará
# el experimento default del notebook, lo que funciona igual de bien.
# mlflow.set_experiment("/Shared/aqi-rf-multinational")

with mlflow.start_run() as run:
    mlflow.log_params(PARAMS)
    mlflow.log_param("data_source", "openaq-multipais")
    mlflow.log_param("n_rows", len(df))
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("n_countries", len(country_mapping))

    rf = RandomForestClassifier(n_jobs=-1, **PARAMS)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    mlflow.log_metric("accuracy_test", acc)
    mlflow.log_metric("f1_macro", f1_macro)
    mlflow.sklearn.log_model(rf, artifact_path="model")

    run_id = run.info.run_id

print(f"\naccuracy_test = {acc:.4f}")
print(f"f1_macro      = {f1_macro:.4f}\n")
print(classification_report(
    y_test, y_pred,
    target_names=[AQI_LABELS[i] for i in sorted(AQI_LABELS)],
    digits=4, zero_division=0,
))

# COMMAND ----------

# MAGIC %md ## 4. Serializar modelo + metadata

# COMMAND ----------

import json
import joblib
import os

# Medianas para imputación en inferencia (lo usa la API en EC2)
NO_IMPUTE = {"lat", "lon", "hour", "day", "month", "year", "dayofweek", "aqi_class"}
medians = {
    c: float(df[c].median())
    for c in numeric_cols
    if c not in NO_IMPUTE and df[c].notna().any()
}

metadata = {
    "feature_names": feature_cols,
    "target": TARGET,
    "class_labels": {str(k): v for k, v in AQI_LABELS.items()},
    "medians": medians,
    "country_encoder": country_mapping,
    "model_type": "RandomForestClassifier",
    "params": PARAMS,
    "metrics": {"accuracy_test": float(acc), "f1_macro": float(f1_macro)},
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "training_source": "databricks",
    "data_source": "openaq",
    "countries": sorted(country_mapping.keys()),
    "mlflow_run_id": run_id,
}

local_model = "/tmp/rf_aqi.pkl"
local_meta = "/tmp/features.json"
joblib.dump(rf, local_model)
with open(local_meta, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Tamaño del .pkl: {round(os.path.getsize(local_model) / 1e6, 2)} MB")
print(f"Tamaño features.json: {round(os.path.getsize(local_meta) / 1e3, 2)} KB")

# COMMAND ----------

# MAGIC %md ## 5. Publicar en S3
# MAGIC
# MAGIC Solo se promueve el modelo si supera el umbral de accuracy (gate de calidad).
# MAGIC La API en EC2 lo descargará al reiniciarse — usa el IAM role asociado.

# COMMAND ----------

import boto3

if acc < MIN_ACCURACY:
    raise ValueError(
        f"accuracy_test={acc:.4f} < umbral {MIN_ACCURACY}. NO se publica el modelo."
    )

aws_key = dbutils.secrets.get(scope="aqi", key="aws_access_key_id")
aws_secret = dbutils.secrets.get(scope="aqi", key="aws_secret_access_key")

s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_key,
    aws_secret_access_key=aws_secret,
    region_name=AWS_REGION,
)

s3.upload_file(local_model, MODEL_BUCKET, "rf_aqi.pkl")
s3.upload_file(local_meta, MODEL_BUCKET, "features.json")

print(f"✅ Modelo publicado en s3://{MODEL_BUCKET}/")
print("\nPara que la EC2 sirva el modelo nuevo:")
print("  1. SSH a la EC2 (o consola web)")
print("  2. docker stop aqi-api && docker rm aqi-api")
print("  3. rm -rf /tmp/aqi-model")
print("  4. docker run -d --name aqi-api -p 8000:8000 \\")
print("       -e S3_BUCKET=" + MODEL_BUCKET + " \\")
print("       -e AWS_REGION=" + AWS_REGION + " \\")
print("       --restart unless-stopped aqi-api")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Listo
# MAGIC
# MAGIC - Modelo entrenado y registrado en **MLflow** (Experiments → `/Shared/aqi-rf-multinational`)
# MAGIC - `.pkl` + `features.json` subidos a S3 con gate de calidad aplicado
# MAGIC - La EC2 está listo para descargar el modelo nuevo al reiniciarse
