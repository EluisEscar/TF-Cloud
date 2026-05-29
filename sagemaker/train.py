"""Script de entrenamiento que ejecuta SageMaker dentro del container SKLearn.

Convenciones de SageMaker:
- Datos de entrada      → /opt/ml/input/data/{channel_name}/
- Modelo de salida      → /opt/ml/model/  (SageMaker lo empaqueta a model.tar.gz)
- Hiperparámetros       → argparse (SageMaker los pasa como --flags)

Lee el Parquet preparado, entrena un RandomForest, sube el .pkl y features.json
directamente al model bucket (en paralelo al model.tar.gz estándar que sube
SageMaker) para que la API en EC2 pueda descargarlos sin descomprimir.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import boto3
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

# Features posibles. La lista real se decide en runtime según qué columnas
# trae el Parquet (porque prepare_data.py dropea las que tienen >70% NaN).
CANDIDATE_NUMERIC = [
    "pm1", "pm10", "relativehumidity", "temperature", "um003",
    "co", "no2", "o3", "so2", "wind_speed", "wind_direction",
    "lat", "lon",
]
TEMPORAL_COLS = ["hour", "day", "month", "year", "dayofweek"]
CATEGORICAL_COL = "country_id"  # se deriva de country_code con LabelEncoder
TARGET = "aqi_class"

# Columnas que no se imputan con mediana (geográficas / temporales / target)
NO_IMPUTE = {"lat", "lon", "hour", "day", "month", "year", "dayofweek", "aqi_class"}

AQI_LABELS = {
    0: "Buena",
    1: "Moderada",
    2: "Dañina para grupos sensibles",
    3: "Dañina",
    4: "Muy dañina",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    p.add_argument("--model-bucket", default=os.environ.get("MODEL_BUCKET", ""))
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=20)
    p.add_argument("--min-samples-leaf", type=int, default=50)
    p.add_argument("--min-accuracy", type=float, default=0.65,
                   help="Gate: si accuracy_test < umbral, no se sube el modelo a S3.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log.info("Args: %s", vars(args))

    # ─── Cargar Parquet ────────────────────────────────────────────────────
    train_dir = Path(args.train)
    parquet_files = list(train_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No hay .parquet en {train_dir}")
    log.info("Leyendo %d archivo(s) Parquet ...", len(parquet_files))
    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    log.info("Shape: %s", df.shape)
    log.info("Columnas: %s", list(df.columns))

    # ─── Label-encode country_code ─────────────────────────────────────────
    if "country_code" not in df.columns:
        raise ValueError("Falta la columna country_code en el dataset")
    encoder = LabelEncoder()
    df[CATEGORICAL_COL] = encoder.fit_transform(df["country_code"].astype(str))
    country_mapping = {
        code: int(idx) for code, idx in zip(encoder.classes_, encoder.transform(encoder.classes_))
    }
    log.info("country_encoder: %s", country_mapping)

    # ─── Decidir feature set real ──────────────────────────────────────────
    numeric_cols = [c for c in CANDIDATE_NUMERIC if c in df.columns]
    feature_cols = numeric_cols + TEMPORAL_COLS + [CATEGORICAL_COL]
    log.info("Features (%d): %s", len(feature_cols), feature_cols)

    X = df[feature_cols].astype(float)
    y = df[TARGET].astype(int)
    log.info("X: %s | y: %s", X.shape, y.shape)
    log.info("Distribución y:\n%s", y.value_counts().sort_index())

    # ─── Train / test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ─── Entrenar ──────────────────────────────────────────────────────────
    log.info("Entrenando RandomForestClassifier ...")
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # ─── Evaluar ───────────────────────────────────────────────────────────
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    log.info("accuracy_test = %.4f", acc)
    log.info("f1_macro      = %.4f", f1m)
    log.info("\n%s", classification_report(
        y_test, y_pred,
        target_names=[AQI_LABELS[i] for i in sorted(AQI_LABELS)],
        digits=4,
        zero_division=0,
    ))

    # ─── Medianas para imputación en inferencia ────────────────────────────
    medians = {
        c: float(df[c].median())
        for c in numeric_cols if c not in NO_IMPUTE and df[c].notna().any()
    }

    # ─── Metadata ──────────────────────────────────────────────────────────
    metadata = {
        "feature_names": feature_cols,
        "target": TARGET,
        "class_labels": {str(k): v for k, v in AQI_LABELS.items()},
        "medians": medians,
        "country_encoder": country_mapping,
        "model_type": "RandomForestClassifier",
        "params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "metrics": {"accuracy_test": float(acc), "f1_macro": float(f1m)},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "training_source": "sagemaker",
        "data_source": "openaq",
        "countries": sorted(country_mapping.keys()),
    }

    # ─── Guardar localmente (SageMaker hace tar.gz de model_dir) ───────────
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    local_pkl = model_dir / "rf_aqi.pkl"
    local_meta = model_dir / "features.json"
    joblib.dump(rf, local_pkl)
    with open(local_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log.info("Guardado en %s y %s", local_pkl, local_meta)

    # ─── Gate de calidad ───────────────────────────────────────────────────
    if acc < args.min_accuracy:
        log.error(
            "accuracy_test=%.4f < umbral %.2f. NO se sube a S3 (modelo viejo se conserva).",
            acc, args.min_accuracy,
        )
        return  # SageMaker termina OK, pero no contaminamos producción

    # ─── Subida directa al model bucket (para que la API EC2 los descargue) ─
    if args.model_bucket:
        s3 = boto3.client("s3")
        log.info("Subiendo a s3://%s/ ...", args.model_bucket)
        s3.upload_file(str(local_pkl), args.model_bucket, "rf_aqi.pkl")
        s3.upload_file(str(local_meta), args.model_bucket, "features.json")
        log.info("✅ Modelo publicado en s3://%s/", args.model_bucket)
    else:
        log.warning("MODEL_BUCKET no definido — modelo solo en SageMaker tar.gz")


if __name__ == "__main__":
    main()
