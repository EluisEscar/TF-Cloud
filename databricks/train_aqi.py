# Databricks notebook source
# MAGIC %md
# MAGIC # Reentrenamiento del modelo de calidad del aire (Almaty)
# MAGIC
# MAGIC Notebook de Databricks que reentrena el `RandomForestClassifier`, registra el
# MAGIC experimento en **MLflow** y publica el modelo en **Hugging Face Hub** para que
# MAGIC la API en Render lo descargue al arrancar.
# MAGIC
# MAGIC **Flujo:** datos (Volume o Supabase) → entrenamiento → MLflow → Hugging Face Hub.
# MAGIC
# MAGIC ## Requisitos previos (una sola vez)
# MAGIC
# MAGIC Configura estos *secrets* en Databricks (CLI: `databricks secrets ...`) en un
# MAGIC scope llamado `aqi`:
# MAGIC - `aqi/hf_token`  → token de escritura de Hugging Face (Settings → Access Tokens).
# MAGIC - `aqi/database_url` → connection string Postgres de Supabase (solo si lees de Supabase).
# MAGIC
# MAGIC Los parámetros (repo de HF, fuente de datos) se controlan con los *widgets* de abajo.

# COMMAND ----------

# MAGIC %pip install -q "scikit-learn>=1.6.1" "huggingface_hub>=0.27" "sqlalchemy>=2.0" "psycopg2-binary>=2.9" pandas joblib
# MAGIC %restart_python

# COMMAND ----------

# Widgets de parámetros (también los lee el Job programado de la Fase C).
dbutils.widgets.text("hf_repo", "TU_USUARIO_HF/aqi-rf", "Repo Hugging Face (user/nombre)")
dbutils.widgets.dropdown("data_source", "volume", ["volume", "supabase"], "Fuente de datos")
dbutils.widgets.text(
    "volume_path",
    "/Volumes/workspace/default/aqi/processed.csv",
    "Ruta del CSV en el Volume (si data_source=volume)",
)

HF_REPO = dbutils.widgets.get("hf_repo")
DATA_SOURCE = dbutils.widgets.get("data_source")
VOLUME_PATH = dbutils.widgets.get("volume_path")

print(f"Repo HF        : {HF_REPO}")
print(f"Fuente de datos: {DATA_SOURCE}")

# COMMAND ----------

# MAGIC %md ## 1. Cargar datos

# COMMAND ----------

import pandas as pd

FEATURE_COLS = [
    "pm10", "relativehumidity", "temperature", "um003",
    "hour", "day", "month", "year", "dayofweek",
    "lat", "lon",
]
TARGET_COL = "aqi_class"
MEDIAN_COLS = ["pm10", "relativehumidity", "temperature", "um003"]

# Etiquetas EPA (mismas que src/preprocessing.py del repo).
AQI_LABELS = {
    0: "Buena",
    1: "Moderada",
    2: "Dañina para grupos sensibles",
    3: "Dañina",
    4: "Muy dañina",
}

if DATA_SOURCE == "supabase":
    # Lee la tabla air_quality (ya contiene aqi_class) desde Supabase.
    from sqlalchemy import create_engine

    database_url = dbutils.secrets.get(scope="aqi", key="database_url")
    engine = create_engine(database_url)
    df = pd.read_sql(
        "SELECT pm10, relativehumidity, temperature, um003, hour, day, "
        "month, year, dayofweek, lat, lon, aqi_class FROM air_quality",
        engine,
    )
else:
    # Lee el CSV procesado subido a un Volume de Unity Catalog.
    df = pd.read_csv(VOLUME_PATH)

print("Filas cargadas:", len(df))
df[[TARGET_COL]].value_counts().sort_index()

# COMMAND ----------

# MAGIC %md ## 2. Preparar X / y y particionar

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = df[FEATURE_COLS].astype(float)
y = df[TARGET_COL].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)
print("Train:", X_train.shape, "| Test:", X_test.shape)

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

mlflow.set_experiment("/Shared/aqi-rf")

with mlflow.start_run() as run:
    mlflow.log_params(PARAMS)
    mlflow.log_param("data_source", DATA_SOURCE)
    mlflow.log_param("n_rows", len(df))

    rf = RandomForestClassifier(n_jobs=-1, **PARAMS)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    mlflow.log_metric("accuracy_test", acc)
    mlflow.log_metric("f1_macro", f1_macro)
    mlflow.sklearn.log_model(rf, artifact_path="model")

    run_id = run.info.run_id

print(f"accuracy_test = {acc:.4f} | f1_macro = {f1_macro:.4f}")
print(classification_report(y_test, y_pred, target_names=[AQI_LABELS[i] for i in sorted(AQI_LABELS)], digits=4))

# COMMAND ----------

# MAGIC %md ## 4. Serializar modelo + metadata (mismo formato que el repo)

# COMMAND ----------

import json
import joblib

medians = {col: float(df[col].median()) for col in MEDIAN_COLS}

metadata = {
    "feature_names": FEATURE_COLS,
    "target": TARGET_COL,
    "class_labels": {str(k): v for k, v in AQI_LABELS.items()},
    "medians": medians,
    "model_type": "RandomForestClassifier",
    "params": PARAMS,
    "metrics": {"accuracy_test": float(acc), "f1_macro": float(f1_macro)},
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "mlflow_run_id": run_id,
}

local_model = "/tmp/rf_aqi.pkl"
local_meta = "/tmp/features.json"
joblib.dump(rf, local_model)
with open(local_meta, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("Tamaño del .pkl:", round(__import__("os").path.getsize(local_model) / 1e6, 2), "MB")

# COMMAND ----------

# MAGIC %md ## 5. Publicar en Hugging Face Hub
# MAGIC
# MAGIC Solo promueve el modelo si supera un umbral mínimo de accuracy (gate de calidad).

# COMMAND ----------

from huggingface_hub import HfApi

MIN_ACCURACY = 0.65  # gate: no publiques un modelo peor que la línea base

if acc < MIN_ACCURACY:
    raise ValueError(
        f"accuracy_test={acc:.4f} < umbral {MIN_ACCURACY}. No se publica el modelo."
    )

hf_token = dbutils.secrets.get(scope="aqi", key="hf_token")
api = HfApi(token=hf_token)
api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True)

api.upload_file(path_or_fileobj=local_model, path_in_repo="rf_aqi.pkl", repo_id=HF_REPO)
api.upload_file(path_or_fileobj=local_meta, path_in_repo="features.json", repo_id=HF_REPO)

print(f"Modelo publicado en https://huggingface.co/{HF_REPO}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Listo
# MAGIC
# MAGIC La API en Render descargará `rf_aqi.pkl` y `features.json` de este repo de HF
# MAGIC al reiniciarse (siempre que tenga la variable de entorno `HF_MODEL_REPO`).
# MAGIC
# MAGIC Para forzar que Render tome el nuevo modelo: en el dashboard de Render →
# MAGIC **Manual Deploy** → *Clear build cache & deploy*, o configura un Deploy Hook.
