"""FastAPI app: clasificación de calidad del aire (modelo multinacional).

Endpoints:
- GET  /              healthcheck
- GET  /model-info    metadata del modelo (features, clases, métricas)
- POST /predict       clasifica una observación
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
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("aqi-api")

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE.parent / "models" / "rf_aqi.pkl"
FEATURES_JSON = HERE.parent / "models" / "features.json"

state: dict[str, Any] = {"model": None, "meta": None}


def resolve_model_files() -> tuple[Path, Path]:
    """Decide de dónde sale el modelo según las env vars (precedencia: S3 > HF > local)."""
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
    title="Air Quality API — Monitor multinacional",
    description=(
        "Clasificación supervisada del nivel de calidad del aire en 10 países "
        "con alta concentración de PM2.5 (Bangladesh, India, Pakistán, Nepal, "
        "Mongolia, Tailandia, Vietnam, Kazajistán, Indonesia, México). "
        "Random Forest entrenado en Databricks sobre mediciones reales de OpenAQ, "
        "expuesto desde una EC2 en AWS. Las clases siguen los breakpoints EPA "
        "para PM2.5."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionInput(BaseModel):
    """Schema flexible: el endpoint usa solo las features que el modelo cargado
    declara en ``feature_names``; lo demás se ignora o se imputa con medianas."""

    pm1: Optional[float] = Field(default=None, description="PM1 (µg/m³)")
    pm10: Optional[float] = Field(default=None, description="PM10 (µg/m³) — modelo legacy")
    relativehumidity: Optional[float] = Field(default=None, description="Humedad relativa (%)")
    temperature: Optional[float] = Field(default=None, description="Temperatura (°C)")
    um003: Optional[float] = Field(default=None, description="Conteo de partículas ≥0.3µm")
    hour: int = Field(..., ge=0, le=23)
    day: int = Field(..., ge=1, le=31)
    month: int = Field(..., ge=1, le=12)
    year: int = Field(..., ge=2020, le=2030)
    dayofweek: int = Field(..., ge=0, le=6, description="0=lunes, 6=domingo")
    lat: float
    lon: float
    country_code: Optional[str] = Field(
        default=None, description="ISO alpha-2 (ej. 'BD', 'IN'). Requerido por el modelo multipaís."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "pm1": 30.0,
                "relativehumidity": 65.0,
                "temperature": 28.0,
                "um003": 3000.0,
                "hour": 14,
                "day": 15,
                "month": 3,
                "year": 2026,
                "dayofweek": 4,
                "lat": 23.73,
                "lon": 90.40,
                "country_code": "BD",
            }
        }
    }


class PredictionOutput(BaseModel):
    aqi_class: int
    aqi_label: str
    probabilities: dict[str, float]


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
    country_encoder: Optional[dict[str, int]] = None
    countries: Optional[list[str]] = None


@app.get("/", response_model=HealthOutput, tags=["health"])
def healthcheck() -> HealthOutput:
    return HealthOutput(status="ok", model_loaded=state["model"] is not None)


@app.get("/model-info", response_model=ModelInfoOutput, tags=["info"])
def model_info() -> ModelInfoOutput:
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
        country_encoder=meta.get("country_encoder"),
        countries=meta.get("countries"),
    )


@app.post("/predict", response_model=PredictionOutput, tags=["predict"])
def predict(payload: PredictionInput) -> PredictionOutput:
    """Clasifica una observación en una de las 5 clases EPA según breakpoints PM2.5."""
    model = state["model"]
    meta = state["meta"]
    if model is None or meta is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    medians: dict[str, float] = meta["medians"]
    feature_names: list[str] = meta["feature_names"]
    raw = payload.model_dump()

    # Traducir country_code → country_id si el modelo lo requiere.
    # País desconocido cae al primer ID del encoder (predicción menos precisa).
    if "country_id" in feature_names:
        encoder = meta.get("country_encoder", {})
        code = raw.get("country_code")
        if code and code in encoder:
            raw["country_id"] = encoder[code]
        else:
            raw["country_id"] = next(iter(encoder.values()), 0)
            log.warning("country_code %r no está en el encoder; usando %s", code, raw["country_id"])

    # Construye la fila en el orden exacto de feature_names; missing → mediana del training.
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
