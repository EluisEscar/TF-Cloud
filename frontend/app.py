"""Frontend Streamlit del sistema de clasificación de calidad del aire en Almaty.

Tabs:
1. Predicción en tiempo real (llama a la API FastAPI).
2. Visualización histórica (consulta Supabase: mapa, línea temporal,
   distribución de clases, estadísticas resumen).
3. Sobre el proyecto.

Configuración:
- API_URL: URL de la FastAPI (default http://localhost:8000).
- SUPABASE_URL, SUPABASE_ANON_KEY: para consultar datos históricos.

Las dos primeras vienen de st.secrets (deploy en Streamlit Cloud) o
de variables de entorno (desarrollo local con .env).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(
    page_title="Calidad del aire - Almaty",
    page_icon="AQ",
    layout="wide",
)

# Carga .env local si existe (no afecta a Streamlit Cloud)
load_dotenv()


# Config


def get_setting(key: str, default: str = "") -> str:
    """Lee primero de variables de entorno, luego de st.secrets si existe."""
    value = os.getenv(key)
    if value:
        return value

    local_secrets = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
    ]
    if not any(path.exists() for path in local_secrets):
        return default

    try:
        if key in st.secrets:
            return st.secrets[key]
    except (FileNotFoundError, KeyError):
        pass
    return default


def normalize_supabase_url(url: str) -> str:
    """Supabase client expects the project URL, not the REST endpoint."""
    return url.strip().rstrip("/").removesuffix("/rest/v1")


API_URL = get_setting("API_URL", "http://localhost:8000").rstrip("/")
SUPABASE_URL = normalize_supabase_url(get_setting("SUPABASE_URL", ""))
SUPABASE_ANON_KEY = get_setting("SUPABASE_ANON_KEY", "")


# Mapa de clase → color para los gráficos
CLASS_COLORS = {
    "Buena": "#2ecc71",
    "Moderada": "#f1c40f",
    "Dañina para grupos sensibles": "#e67e22",
    "Dañina": "#e74c3c",
    "Muy dañina": "#8e44ad",
}

CLASS_ORDER = [
    "Buena",
    "Moderada",
    "Dañina para grupos sensibles",
    "Dañina",
    "Muy dañina",
]




# Cliente Supabase (lazy)


@st.cache_resource(show_spinner=False)
def get_supabase_client():
    """Crea el cliente Supabase. Devuelve None si faltan credenciales."""
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None
    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo conectar a Supabase: {exc}")
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_recent(limit: int = 5000) -> pd.DataFrame:
    """Trae las últimas N mediciones desde Supabase, ordenadas por fecha."""
    client = get_supabase_client()
    if client is None:
        return pd.DataFrame()
    try:
        res = (
            client.table("air_quality")
            .select("datetime,location_id,name,lat,lon,pm25,aqi_class,aqi_label")
            .order("datetime", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo consultar Supabase: {exc}")
        return pd.DataFrame()
    df = pd.DataFrame(res.data)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stations() -> pd.DataFrame:
    """Devuelve las estaciones únicas con lat/lon (para el mapa)."""
    client = get_supabase_client()
    if client is None:
        return pd.DataFrame()
    try:
        res = (
            client.table("air_quality")
            .select("location_id,name,lat,lon")
            .limit(50000)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo consultar estaciones en Supabase: {exc}")
        return pd.DataFrame()
    df = pd.DataFrame(res.data)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["location_id"])


# API


def call_predict(payload: dict) -> dict | None:
    """Llama POST /predict de la FastAPI. Devuelve el JSON o None si falla."""
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"Error llamando a la API: {exc}")
        return None


def call_model_info() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


# UI


st.title("Calidad del aire - Almaty")
st.caption(
    "Sistema de clasificación supervisada basado en Random Forest. "
    "Dataset: OpenAQ Almaty (2020-2026) · Curso Cloud Computing · USIL."
)

tab_pred, tab_hist, tab_about = st.tabs(
    ["Prediccion", "Historico", "Sobre el proyecto"]
)


# Tab 1: Prediccion

with tab_pred:
    st.subheader("Predecir la clase de calidad del aire")
    st.write(
        "Ingresa las condiciones ambientales y temporales. Los campos vacíos "
        "se imputan con la mediana del training."
    )

    now = datetime.utcnow()
    st.caption(
        f"Por defecto la predicción se hace para **ahora** "
        f"(`{now.strftime('%Y-%m-%d %H:%M UTC')}`). "
        f"Puedes simular otra fecha desplegando *Ajustar fecha/hora*."
    )

    with st.form("pred_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Mediciones ambientales**")
            pm10 = st.number_input("PM10 (µg/m³)", value=35.0, min_value=0.0, step=1.0)
            humidity = st.number_input("Humedad relativa (%)", value=60.0, min_value=0.0, max_value=100.0)
            temperature = st.number_input("Temperatura (°C)", value=5.0, step=0.5)
            um003 = st.number_input("Partículas ≥0.3µm (um003)", value=2000.0, min_value=0.0, step=100.0)
        with col2:
            st.markdown("**Dónde**")
            lat = st.number_input("Latitud", value=43.2520, format="%.6f")
            lon = st.number_input("Longitud", value=76.9285, format="%.6f")
            st.caption("Centro de Almaty: 43.25, 76.93")

        with st.expander("⚙️ Ajustar fecha/hora (opcional)", expanded=False):
            st.caption(
                "El modelo aprendió patrones estacionales y horarios. "
                "Por defecto se usa la fecha/hora actual (UTC); modifica los "
                "valores para simular otro momento."
            )
            tcol1, tcol2, tcol3 = st.columns(3)
            with tcol1:
                hour = st.slider("Hora (UTC)", 0, 23, now.hour)
                dayofweek = st.selectbox(
                    "Día de la semana",
                    options=list(range(7)),
                    format_func=lambda i: ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"][i],
                    index=now.weekday(),
                )
            with tcol2:
                day = st.slider("Día del mes", 1, 31, now.day)
                month = st.slider("Mes", 1, 12, now.month)
            with tcol3:
                year = st.number_input(
                    "Año", value=now.year, min_value=2020, max_value=2030
                )

        submitted = st.form_submit_button("🔮 Predecir", use_container_width=True, type="primary")

    if submitted:
        payload = {
            "pm10": pm10,
            "relativehumidity": humidity,
            "temperature": temperature,
            "um003": um003,
            "hour": int(hour),
            "day": int(day),
            "month": int(month),
            "year": int(year),
            "dayofweek": int(dayofweek),
            "lat": float(lat),
            "lon": float(lon),
        }
        with st.spinner("Llamando al modelo..."):
            result = call_predict(payload)

        if result:
            label = result["aqi_label"]
            color = CLASS_COLORS.get(label, "#34495e")
            st.markdown(
                f"""
                <div style="
                    background-color:{color};
                    padding:1.5rem;
                    border-radius:0.75rem;
                    text-align:center;
                    color:white;
                    margin-top:1rem;
                ">
                    <h2 style="margin:0;color:white;">Clase {result['aqi_class']}: {label}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Probabilidad por clase")
            probs = result["probabilities"]
            prob_df = pd.DataFrame(
                {"clase": list(probs.keys()), "probabilidad": list(probs.values())}
            )
            prob_df["clase"] = pd.Categorical(prob_df["clase"], categories=CLASS_ORDER, ordered=True)
            prob_df = prob_df.sort_values("clase")
            fig = px.bar(
                prob_df,
                x="clase",
                y="probabilidad",
                color="clase",
                color_discrete_map=CLASS_COLORS,
                text="probabilidad",
            )
            fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
            fig.update_layout(showlegend=False, yaxis_tickformat=".0%", height=350)
            st.plotly_chart(fig, use_container_width=True)


# Tab 2: Historico

with tab_hist:
    st.subheader("Datos históricos — Almaty")

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.warning(
            "Faltan credenciales de Supabase (SUPABASE_URL / SUPABASE_ANON_KEY). "
            "Configúralas en `.env` o `secrets.toml` para ver el histórico."
        )
    else:
        with st.spinner("Consultando Supabase..."):
            df = fetch_recent(limit=5000)
            stations = fetch_stations()

        if df.empty:
            st.info("La tabla `air_quality` está vacía. Carga datos con `src/upload_to_supabase.py`.")
        else:
            # Estadísticas resumen
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mediciones cargadas", f"{len(df):,}")
            c2.metric("PM2.5 promedio", f"{df['pm25'].mean():.1f} µg/m³")
            c3.metric("PM2.5 máximo", f"{df['pm25'].max():.1f} µg/m³")
            c4.metric("Estaciones únicas", f"{stations['location_id'].nunique()}")

            st.markdown("---")

            # Mapa de estaciones
            col_map, col_dist = st.columns([3, 2])
            with col_map:
                st.markdown("#### Estaciones de monitoreo")
                if not stations.empty:
                    st.map(
                        stations[["lat", "lon"]].rename(columns={"lat": "latitude", "lon": "longitude"}),
                        zoom=10,
                    )
                else:
                    st.info("Sin estaciones disponibles.")

            with col_dist:
                st.markdown("#### Distribución de clases (muestra reciente)")
                dist = (
                    df["aqi_label"].value_counts()
                    .reindex(CLASS_ORDER).fillna(0).reset_index()
                )
                dist.columns = ["clase", "count"]
                fig = px.bar(
                    dist, x="clase", y="count",
                    color="clase",
                    color_discrete_map=CLASS_COLORS,
                )
                fig.update_layout(showlegend=False, height=350, xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Evolución temporal
            st.markdown("#### 📈 Evolución de PM2.5")
            df_sorted = df.sort_values("datetime")
            # Agregamos por día para no saturar
            daily = (
                df_sorted.set_index("datetime")["pm25"]
                .resample("D").mean().dropna().reset_index()
            )
            fig_line = px.line(
                daily, x="datetime", y="pm25",
                labels={"datetime": "Fecha", "pm25": "PM2.5 (µg/m³)"},
            )
            fig_line.add_hline(y=12, line_dash="dash", line_color="#2ecc71",
                               annotation_text="Buena (≤12)", annotation_position="right")
            fig_line.add_hline(y=35.4, line_dash="dash", line_color="#f39c12",
                               annotation_text="Moderada (≤35.4)", annotation_position="right")
            fig_line.add_hline(y=55.4, line_dash="dash", line_color="#e74c3c",
                               annotation_text="Dañina (≤55.4)", annotation_position="right")
            fig_line.update_layout(height=400)
            st.plotly_chart(fig_line, use_container_width=True)

            with st.expander("Ver datos crudos (últimas 100 filas)"):
                st.dataframe(df.head(100), use_container_width=True)


# Tab 3: Sobre el proyecto

with tab_about:
    st.subheader("Sobre el proyecto")

    st.markdown(
        """
        ### Objetivo
        MVP de un sistema de clasificacion de calidad del aire en **Almaty,
        Kazajistan**, basado en Machine Learning supervisado, desplegado en la
        nube. Trabajo final del curso de **Cloud Computing** de la **USIL**.

        ### Dataset
        - **Fuente:** [Almaty Air Quality History](https://www.kaggle.com/datasets/fichka/almaty-air-quality-history) (Kaggle)
        - **Origen:** OpenAQ (mediciones de estaciones AirNow, Clarity, AirGradient)
        - **Tamano:** ~545K registros - 2020-04 -> 2026-01 - 146 estaciones

        ### Variable objetivo
        Clasificacion de PM2.5 (ug/m3) en 5 clases segun breakpoints EPA:

        | Clase | Rango | Etiqueta |
        |:----:|---|---|
        | 0 | 0 - 12 | Buena |
        | 1 | 12.1 - 35.4 | Moderada |
        | 2 | 35.5 - 55.4 | Danina para grupos sensibles |
        | 3 | 55.5 - 150.4 | Danina |
        | 4 | 150.5+ | Muy danina |

        ### Arquitectura
        ```
        Kaggle CSV
            ->
        EDA + limpieza (pandas en Jupyter)
            ->
        Random Forest (scikit-learn) -> models/rf_aqi.pkl
            ->
        Supabase (PostgreSQL) <-> FastAPI (Render)
            -> Streamlit Cloud
        ```

        ### Stack
        - **Python 3.11**, pandas, scikit-learn
        - **FastAPI + Uvicorn** (API REST, deploy en Render)
        - **Streamlit** (frontend, deploy en Streamlit Community Cloud)
        - **Supabase** (PostgreSQL gestionado, datos historicos)
        - **Docker** (contenedor de la API)
        """
    )

    info = call_model_info()
    if info:
        st.markdown("### Modelo en produccion")
        c1, c2, c3 = st.columns(3)
        c1.metric("Tipo", info["model_type"])
        c2.metric("Accuracy (test)", f"{info['metrics']['accuracy_test']:.2%}")
        c3.metric("Features", len(info["feature_names"]))
        with st.expander("Ver metadata completa"):
            st.json(info)
    else:
        st.info(
            f"No se pudo conectar a la API en `{API_URL}` para mostrar metadata del modelo."
        )

# Footer

st.markdown("---")
st.caption(
    f"API: `{API_URL}` · "
    f"Supabase: `{'configurado' if SUPABASE_URL else 'no configurado'}`"
)
