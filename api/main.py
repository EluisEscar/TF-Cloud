"""FastAPI app: clasificación de calidad del aire en Almaty.

Endpoints:
- GET  /              → healthcheck
- GET  /model-info    → metadata del modelo (features, clases, métricas)
- POST /predict       → clasifica una observación

El modelo y su metadata se cargan al arrancar la app (cold start).
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("aqi-api")

# Rutas del modelo local (bundleado en el Docker image como fallback).
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE.parent / "models" / "rf_aqi.pkl"
FEATURES_JSON = HERE.parent / "models" / "features.json"

# Estado global del modelo (se llena en el lifespan)
state: dict[str, Any] = {"model": None, "meta": None}


def resolve_model_files() -> tuple[Path, Path]:
    """Decide de dónde sale el modelo. Precedencia:

    1. ``S3_BUCKET`` definida → descarga de un bucket S3 (despliegue en AWS EC2).
       ``AWS_REGION`` opcional (defecto ``us-east-1``). Las credenciales las
       resuelve ``boto3`` automáticamente: IAM role en EC2, env vars o
       ``~/.aws/credentials`` en local.
    2. ``HF_MODEL_REPO`` definida → descarga del repo de Hugging Face
       (formato ``usuario/aqi-rf``). ``HF_TOKEN`` solo si el repo es privado.
    3. Sin env vars → usa los archivos bundleados en la imagen Docker
       (comportamiento original de la Fase 5).
    """
    bucket = os.getenv("S3_BUCKET")
    if bucket:
        import boto3

        region = os.getenv("AWS_REGION", "us-east-1")
        log.info("Descargando modelo desde S3: s3://%s (region=%s)", bucket, region)
        s3 = boto3.client("s3", region_name=region)
        cache_dir = Path("/tmp/aqi-model")
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "rf_aqi.pkl"
        features_path = cache_dir / "features.json"
        s3.download_file(bucket, "rf_aqi.pkl", str(model_path))
        s3.download_file(bucket, "features.json", str(features_path))
        return model_path, features_path

    repo = os.getenv("HF_MODEL_REPO")
    if not repo:
        return MODEL_PATH, FEATURES_JSON

    from huggingface_hub import hf_hub_download

    token = os.getenv("HF_TOKEN") or None
    log.info("Descargando modelo desde Hugging Face: %s", repo)
    model_path = Path(hf_hub_download(repo_id=repo, filename="rf_aqi.pkl", token=token))
    features_path = Path(
        hf_hub_download(repo_id=repo, filename="features.json", token=token)
    )
    return model_path, features_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo y la metadata al arrancar."""
    model_path, features_path = resolve_model_files()
    log.info("Cargando modelo desde %s ...", model_path)
    if not model_path.exists():
        raise RuntimeError(f"No existe el modelo: {model_path}")
    if not features_path.exists():
        raise RuntimeError(f"No existe la metadata: {features_path}")

    state["model"] = joblib.load(model_path)
    with features_path.open(encoding="utf-8") as f:
        state["meta"] = json.load(f)
    log.info(
        "Modelo cargado: %s | features=%d | accuracy_test=%.4f",
        state["meta"]["model_type"],
        len(state["meta"]["feature_names"]),
        state["meta"]["metrics"]["accuracy_test"],
    )
    yield
    log.info("Apagando API.")


app = FastAPI(
    title="Almaty Air Quality API",
    description=(
        "Clasificación supervisada del nivel de calidad del aire en Almaty "
        "(Kazajistán) usando un Random Forest entrenado sobre 500K+ "
        "mediciones de OpenAQ (2020-2026). Las clases siguen los breakpoints "
        "EPA para PM2.5."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP: abierto. Restringir a la URL de Streamlit en prod.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ───────────────────────────────────────────────────────────────


class PredictionInput(BaseModel):
    """Entrada para /predict. Todas las features son opcionales:
    si falta una numérica ambiental se imputa con la mediana del training.
    Las temporales (hour, day, etc.) y lat/lon son obligatorias en la práctica
    pero se aceptan opcionales para flexibilidad."""

    pm10: Optional[float] = Field(default=None, description="PM10 (µg/m³)")
    relativehumidity: Optional[float] = Field(default=None, description="Humedad relativa (%)")
    temperature: Optional[float] = Field(default=None, description="Temperatura (°C)")
    um003: Optional[float] = Field(default=None, description="Conteo de partículas ≥0.3µm")
    hour: int = Field(..., ge=0, le=23, description="Hora del día (0-23, UTC)")
    day: int = Field(..., ge=1, le=31, description="Día del mes (1-31)")
    month: int = Field(..., ge=1, le=12, description="Mes (1-12)")
    year: int = Field(..., ge=2020, le=2030, description="Año")
    dayofweek: int = Field(..., ge=0, le=6, description="Día de semana (0=lunes, 6=domingo)")
    lat: float = Field(..., description="Latitud de la estación")
    lon: float = Field(..., description="Longitud de la estación")

    model_config = {
        "json_schema_extra": {
            "example": {
                "pm10": 35.0,
                "relativehumidity": 70.0,
                "temperature": -5.0,
                "um003": 2500.0,
                "hour": 12,
                "day": 15,
                "month": 1,
                "year": 2025,
                "dayofweek": 2,
                "lat": 43.25,
                "lon": 76.93,
            }
        }
    }


class PredictionOutput(BaseModel):
    aqi_class: int = Field(..., description="Clase predicha (0-4)")
    aqi_label: str = Field(..., description="Etiqueta humana de la clase")
    probabilities: dict[str, float] = Field(
        ..., description="Probabilidad por cada clase (label → prob)"
    )


class HealthOutput(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoOutput(BaseModel):
    model_type: str
    feature_names: list[str]
    class_labels: dict[str, str]
    medians: dict[str, float]
    metrics: dict[str, float]
    n_train: int
    n_test: int


# ─── Endpoints ─────────────────────────────────────────────────────────────


@app.get("/", response_model=HealthOutput, tags=["health"])
def healthcheck() -> HealthOutput:
    """Healthcheck básico. Render lo usa para verificar que la app esté viva."""
    return HealthOutput(
        status="ok",
        model_loaded=state["model"] is not None,
    )


@app.get("/model-info", response_model=ModelInfoOutput, tags=["info"])
def model_info() -> ModelInfoOutput:
    """Devuelve los metadatos del modelo: features esperadas, clases, métricas."""
    meta = state["meta"]
    if meta is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return ModelInfoOutput(
        model_type=meta["model_type"],
        feature_names=meta["feature_names"],
        class_labels=meta["class_labels"],
        medians=meta["medians"],
        metrics=meta["metrics"],
        n_train=meta["n_train"],
        n_test=meta["n_test"],
    )


@app.post("/predict", response_model=PredictionOutput, tags=["predict"])
def predict(payload: PredictionInput) -> PredictionOutput:
    """Clasifica una observación en una de las 5 clases EPA."""
    model = state["model"]
    meta = state["meta"]
    if model is None or meta is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    # Construye la fila en el orden exacto de features que vio el entrenamiento.
    medians: dict[str, float] = meta["medians"]
    feature_names: list[str] = meta["feature_names"]
    raw = payload.model_dump()
    row = {
        name: (raw[name] if raw.get(name) is not None else medians.get(name, 0.0))
        for name in feature_names
    }
    X = pd.DataFrame([row], columns=feature_names)

    try:
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
    except Exception as exc:  # noqa: BLE001
        log.exception("Error en predicción")
        raise HTTPException(status_code=500, detail=f"Predicción falló: {exc}") from exc

    labels: dict[str, str] = meta["class_labels"]
    probabilities = {labels[str(i)]: float(p) for i, p in enumerate(proba)}

    return PredictionOutput(
        aqi_class=pred,
        aqi_label=labels[str(pred)],
        probabilities=probabilities,
    )
