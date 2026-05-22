"""Reutilizable preprocessing del dataset Almaty Air Quality.

Estas funciones se usan tanto desde el notebook 01_eda.ipynb como desde
el script que sube los datos a Supabase y, eventualmente, desde la API.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Breakpoints EPA para PM2.5 (µg/m³) → 5 clases de calidad del aire.
# Formato: (low, high, class_id, label)
AQI_BREAKPOINTS: list[tuple[float, float, int, str]] = [
    (0.0, 12.0, 0, "Buena"),
    (12.1, 35.4, 1, "Moderada"),
    (35.5, 55.4, 2, "Dañina para grupos sensibles"),
    (55.5, 150.4, 3, "Dañina"),
    (150.5, float("inf"), 4, "Muy dañina"),
]

AQI_LABELS: dict[int, str] = {row[2]: row[3] for row in AQI_BREAKPOINTS}

# Columnas de medición ambiental presentes en el dataset crudo
MEASUREMENT_COLS: list[str] = [
    "pm25",
    "pm1",
    "pm10",
    "relativehumidity",
    "temperature",
    "um003",
]


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    """Carga el CSV crudo de Kaggle sin transformar nada."""
    return pd.read_csv(csv_path)


def clean_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica del DataFrame de mediciones.

    - Parsea ``datetime`` a pandas datetime (UTC).
    - Elimina filas con ``pm25`` nulo o no positivo.
    - Elimina duplicados exactos (location_id, datetime).
    """
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    out = out.dropna(subset=["datetime", "pm25"])
    out = out[out["pm25"] > 0]
    out["datetime"] = out["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    out = out.drop_duplicates(subset=["location_id", "datetime"], keep="first")
    return out.reset_index(drop=True)


def extract_temporal_features(
    df: pd.DataFrame, dt_col: str = "datetime"
) -> pd.DataFrame:
    """Agrega features temporales derivadas de la columna datetime."""
    out = df.copy()
    dt = out[dt_col]
    out["hour"] = dt.dt.hour.astype("int8")
    out["day"] = dt.dt.day.astype("int8")
    out["month"] = dt.dt.month.astype("int8")
    out["year"] = dt.dt.year.astype("int16")
    out["dayofweek"] = dt.dt.dayofweek.astype("int8")
    return out


def classify_pm25(pm25: float) -> int:
    """Devuelve la clase AQI (0-4) para un valor de PM2.5."""
    if pm25 <= 12.0:
        return 0
    if pm25 <= 35.4:
        return 1
    if pm25 <= 55.4:
        return 2
    if pm25 <= 150.4:
        return 3
    return 4


def add_aqi_class(df: pd.DataFrame, pm25_col: str = "pm25") -> pd.DataFrame:
    """Agrega ``aqi_class`` (0-4) y ``aqi_label`` derivados de PM2.5."""
    out = df.copy()
    out["aqi_class"] = (
        out[pm25_col].astype(float).apply(classify_pm25).astype("int8")
    )
    out["aqi_label"] = out["aqi_class"].map(AQI_LABELS)
    return out


def impute_numeric_with_median(
    df: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Imputa columnas numéricas con su mediana.

    Devuelve el DataFrame imputado y un diccionario {col: mediana} para
    poder replicar la imputación en producción.
    """
    out = df.copy()
    medians: dict[str, float] = {}
    for col in cols:
        if col not in out.columns:
            continue
        med = float(out[col].median())
        medians[col] = med
        out[col] = out[col].fillna(med)
    return out, medians


def preprocess_pipeline(
    df: pd.DataFrame,
    impute_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Pipeline completo: limpieza + features temporales + target + imputación.

    Returns
    -------
    processed : DataFrame listo para EDA y entrenamiento.
    medians : medianas usadas para imputación (para reutilizar en la API).
    """
    if impute_cols is None:
        impute_cols = ["pm10", "relativehumidity", "temperature", "um003"]

    out = clean_measurements(df)
    out = extract_temporal_features(out)
    out = add_aqi_class(out)
    out, medians = impute_numeric_with_median(out, impute_cols)
    return out, medians
