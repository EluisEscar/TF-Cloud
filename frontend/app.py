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


# Centroide aproximado (lat, lon) de la principal ciudad contaminada por país
# Se usa para pre-rellenar la entrada de Predicción al elegir país en el dropdown.
COUNTRY_CENTROIDS: dict[str, tuple[str, float, float]] = {
    "BD": ("Dhaka, Bangladesh", 23.7300, 90.4000),
    "ID": ("Jakarta, Indonesia", -6.2100, 106.8500),
    "IN": ("Delhi, India", 28.6100, 77.2100),
    "KZ": ("Almaty, Kazajistán", 43.2500, 76.9300),
    "MN": ("Ulaanbaatar, Mongolia", 47.9100, 106.9200),
    "MX": ("Ciudad de México, México", 19.4300, -99.1300),
    "NP": ("Kathmandu, Nepal", 27.7100, 85.3200),
    "PK": ("Lahore, Pakistán", 31.5500, 74.3400),
    "TH": ("Bangkok, Tailandia", 13.7600, 100.5000),
    "VN": ("Hanoi, Vietnam", 21.0300, 105.8300),
}




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
    """Trae las últimas N mediciones desde Supabase, ordenadas por fecha.

    Supabase/PostgREST limita el response a 1000 filas por request, así que
    paginamos con .range() hasta alcanzar el ``limit`` deseado.
    """
    client = get_supabase_client()
    if client is None:
        return pd.DataFrame()
    page_size = 1000
    rows: list[dict] = []
    try:
        for start in range(0, limit, page_size):
            end = min(start + page_size - 1, limit - 1)
            res = (
                client.table("air_quality")
                .select("datetime,location_id,name,lat,lon,pm25,aqi_class,aqi_label,country_code")
                .order("datetime", desc=True)
                .range(start, end)
                .execute()
            )
            if not res.data:
                break
            rows.extend(res.data)
            if len(res.data) < page_size:
                break
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo consultar Supabase: {exc}")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stations() -> pd.DataFrame:
    """Devuelve las estaciones únicas con lat/lon (para el mapa).

    Supabase limita a 1000 filas por request, así que paginamos con
    .range() ordenando por id (orden estable y determinístico — sin
    .order() las páginas pueden traer filas repetidas o saltadas).
    """
    client = get_supabase_client()
    if client is None:
        return pd.DataFrame()
    page_size = 1000
    max_rows = 200_000  # techo de seguridad
    rows: list[dict] = []
    try:
        for start in range(0, max_rows, page_size):
            end = start + page_size - 1
            res = (
                client.table("air_quality")
                .select("location_id,name,lat,lon,country_code")
                .order("id")
                .range(start, end)
                .execute()
            )
            if not res.data:
                break
            rows.extend(res.data)
            if len(res.data) < page_size:
                break
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo consultar estaciones en Supabase: {exc}")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
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


@st.cache_data(ttl=300, show_spinner=False)
def call_model_info() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


# UI


st.title("Calidad del aire - Monitor multinacional")
st.caption(
    "Sistema de clasificación supervisada basado en Random Forest. "
    "Dataset: OpenAQ — 10 países con alta concentración de PM2.5. "
    "Curso Cloud Computing · USIL."
)

tab_pred, tab_hist, tab_about = st.tabs(
    ["Prediccion", "Historico", "Sobre el proyecto"]
)


# Tab 1: Prediccion

with tab_pred:
    st.subheader("Predecir la clase de calidad del aire")

    # El modelo en producción puede ser el original (Almaty, pm10) o el
    # multinacional (pm1 + country_code). Adaptamos el formulario al schema
    # real que reporta /model-info.
    model_info = call_model_info()
    feature_names = (model_info or {}).get("feature_names", [])
    countries: list[str] = (model_info or {}).get("countries") or []
    uses_pm1 = "pm1" in feature_names
    uses_country = "country_id" in feature_names and bool(countries)

    if uses_country:
        st.write(
            "Selecciona el país, ingresa las condiciones ambientales y "
            "temporales. Los campos vacíos se imputan con la mediana del training."
        )
    else:
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

    # Dropdown de país FUERA del form para que cambiar país actualice lat/lon
    # en vivo (los widgets dentro de st.form no rerendean hasta el submit).
    selected_country = None
    default_lat, default_lon, default_loc_label = 43.2520, 76.9285, "Centro de Almaty: 43.25, 76.93"
    if uses_country:
        # Filtra a los países que aparecen tanto en el modelo como en nuestro mapa.
        options = [c for c in countries if c in COUNTRY_CENTROIDS] or countries
        default_idx = options.index("BD") if "BD" in options else 0
        selected_country = st.selectbox(
            "País",
            options=options,
            index=default_idx,
            format_func=lambda c: COUNTRY_CENTROIDS.get(c, (c, 0, 0))[0] if c in COUNTRY_CENTROIDS else c,
        )
        if selected_country in COUNTRY_CENTROIDS:
            label, default_lat, default_lon = COUNTRY_CENTROIDS[selected_country]
            default_loc_label = f"Centroide: {label} ({default_lat}, {default_lon})"

    with st.form("pred_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Mediciones ambientales**")
            if uses_pm1:
                pm_value = st.number_input(
                    "PM1 (µg/m³)", value=30.0, min_value=0.0, step=1.0,
                    help="Partículas ≤1 µm. Más finas que PM2.5 — el modelo las usa como predictor.",
                )
            else:
                pm_value = st.number_input(
                    "PM10 (µg/m³)", value=35.0, min_value=0.0, step=1.0,
                )
            humidity = st.number_input("Humedad relativa (%)", value=60.0, min_value=0.0, max_value=100.0)
            temperature = st.number_input("Temperatura (°C)", value=25.0, step=0.5)
            um003 = st.number_input("Partículas ≥0.3µm (um003)", value=2000.0, min_value=0.0, step=100.0)
        with col2:
            st.markdown("**Dónde**")
            st.text_input("Latitud", value=f"{default_lat:.6f}", disabled=True)
            st.text_input("Longitud", value=f"{default_lon:.6f}", disabled=True)
            st.caption(default_loc_label)
            # Mantenemos los valores reales para enviar al modelo
            lat = default_lat
            lon = default_lon

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
        # Envía el campo de PM correcto según el modelo cargado
        if uses_pm1:
            payload["pm1"] = pm_value
        else:
            payload["pm10"] = pm_value
        if uses_country and selected_country:
            payload["country_code"] = selected_country
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


# Tab 2: Historico (Dashboard)

with tab_hist:
    st.subheader("Dashboard de calidad del aire")

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.warning(
            "Faltan credenciales de Supabase (SUPABASE_URL / SUPABASE_ANON_KEY). "
            "Configúralas en `.env` o `secrets.toml` para ver el histórico."
        )
    else:
        with st.spinner("Cargando histórico de Supabase..."):
            df_full = fetch_recent(limit=100_000)
            stations = fetch_stations()

        if df_full.empty:
            st.info("La tabla `air_quality` está vacía. Carga datos con `ml_pipeline/upload_to_supabase.py`.")
        else:
            # ── Filtros ──────────────────────────────────────────────────
            df_full["country_code"] = df_full["country_code"].fillna("?")
            df_full["date"] = df_full["datetime"].dt.date
            min_date, max_date = df_full["date"].min(), df_full["date"].max()
            available_countries = sorted(df_full["country_code"].unique())

            fcol1, fcol2, fcol3 = st.columns([2, 2, 2])
            with fcol1:
                date_range = st.date_input(
                    "Rango de fechas",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date, end_date = min_date, max_date
            with fcol2:
                selected_countries = st.multiselect(
                    "Países",
                    options=available_countries,
                    default=available_countries,
                    format_func=lambda c: COUNTRY_CENTROIDS.get(c, (c, 0, 0))[0] if c in COUNTRY_CENTROIDS else c,
                )
            with fcol3:
                metric_view = st.selectbox(
                    "Granularidad temporal",
                    options=["Diaria", "Semanal", "Por hora"],
                    index=0,
                    help="Cómo agregar los datos en el gráfico de tendencia.",
                )

            # Aplica filtros
            mask = (
                (df_full["date"] >= start_date)
                & (df_full["date"] <= end_date)
                & (df_full["country_code"].isin(selected_countries or available_countries))
            )
            df = df_full.loc[mask].copy()

            if df.empty:
                st.warning("No hay datos en el rango/países seleccionados.")
                st.stop()

            # ── KPI cards ────────────────────────────────────────────────
            avg_pm = df["pm25"].mean()
            peak_pm = df["pm25"].max()
            peak_row = df.loc[df["pm25"].idxmax()]
            dominant_class = df["aqi_label"].mode().iloc[0]
            n_active_stations = df["location_id"].nunique()

            # Color del avg según breakpoint EPA
            if avg_pm <= 12:
                avg_color, avg_label = "#2ecc71", "Buena"
            elif avg_pm <= 35.4:
                avg_color, avg_label = "#f1c40f", "Moderada"
            elif avg_pm <= 55.4:
                avg_color, avg_label = "#e67e22", "Sensibles"
            elif avg_pm <= 150.4:
                avg_color, avg_label = "#e74c3c", "Dañina"
            else:
                avg_color, avg_label = "#8e44ad", "Muy dañina"

            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "PM2.5 promedio",
                f"{avg_pm:.1f} µg/m³",
                delta=avg_label,
                delta_color="off",
            )
            k2.metric(
                "Pico de polución",
                f"{peak_pm:.1f} µg/m³",
                delta=f"{peak_row['country_code']} · {peak_row['datetime'].strftime('%d %b')}",
                delta_color="off",
            )
            k3.metric("Clase dominante", dominant_class)
            k4.metric("Estaciones activas", n_active_stations)

            st.markdown("---")

            # ── Tendencia + Heatmap ──────────────────────────────────────
            col_trend, col_heat = st.columns([3, 2])

            with col_trend:
                st.markdown("#### Tendencia de PM2.5")
                rule = {"Diaria": "D", "Semanal": "W", "Por hora": "h"}[metric_view]
                trend = (
                    df.set_index("datetime")["pm25"]
                    .resample(rule).mean().dropna().reset_index()
                )
                fig_trend = px.area(
                    trend, x="datetime", y="pm25",
                    labels={"datetime": "Fecha", "pm25": "PM2.5 (µg/m³)"},
                )
                fig_trend.update_traces(line_color="#3498db", fillcolor="rgba(52,152,219,0.2)")
                fig_trend.add_hline(y=12, line_dash="dash", line_color="#2ecc71",
                                    annotation_text="Buena", annotation_position="right")
                fig_trend.add_hline(y=35.4, line_dash="dash", line_color="#f1c40f",
                                    annotation_text="Moderada", annotation_position="right")
                fig_trend.add_hline(y=55.4, line_dash="dash", line_color="#e74c3c",
                                    annotation_text="Dañina", annotation_position="right")
                fig_trend.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_trend, use_container_width=True)

            with col_heat:
                st.markdown("#### Hora vs. día (PM2.5 promedio)")
                pivot = (
                    df.assign(hour=df["datetime"].dt.hour,
                              dayofweek=df["datetime"].dt.dayofweek)
                    .pivot_table(values="pm25", index="dayofweek", columns="hour", aggfunc="mean")
                )
                # Reindexa días para que vayan Lun→Dom
                pivot = pivot.reindex(range(7))
                day_labels = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
                fig_heat = px.imshow(
                    pivot.values,
                    labels=dict(x="Hora", y="Día", color="PM2.5"),
                    x=list(pivot.columns),
                    y=day_labels,
                    color_continuous_scale=[
                        (0.00, "#2ecc71"),  # verde
                        (0.20, "#f1c40f"),  # amarillo
                        (0.45, "#e67e22"),  # naranja
                        (0.70, "#e74c3c"),  # rojo
                        (1.00, "#8e44ad"),  # morado
                    ],
                    aspect="auto",
                )
                fig_heat.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_heat, use_container_width=True)

            st.markdown("---")

            # ── Mapa + Distribución ──────────────────────────────────────
            col_map, col_dist = st.columns([3, 2])

            with col_map:
                st.markdown("#### Estaciones activas")
                stations_filtered = stations[stations["country_code"].isin(selected_countries or available_countries)]
                if not stations_filtered.empty:
                    st.map(
                        stations_filtered[["lat", "lon"]].rename(columns={"lat": "latitude", "lon": "longitude"}),
                        zoom=2,
                    )
                else:
                    st.info("Sin estaciones en la selección.")

            with col_dist:
                st.markdown("#### Distribución de clases")
                dist = (
                    df["aqi_label"].value_counts()
                    .reindex(CLASS_ORDER).fillna(0).reset_index()
                )
                dist.columns = ["clase", "count"]
                fig_dist = px.bar(
                    dist, x="clase", y="count",
                    color="clase",
                    color_discrete_map=CLASS_COLORS,
                )
                fig_dist.update_layout(
                    showlegend=False, height=350,
                    xaxis_tickangle=-30,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            st.markdown("---")

            # ── Tabla + Export ──────────────────────────────────────────
            st.markdown("#### Registros recientes")
            recent = df.sort_values("datetime", ascending=False).head(100)[
                ["datetime", "country_code", "name", "pm25", "aqi_label"]
            ].rename(columns={
                "datetime": "Fecha",
                "country_code": "País",
                "name": "Estación",
                "pm25": "PM2.5",
                "aqi_label": "Clase",
            })
            st.dataframe(recent, use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Exportar CSV (selección actual)",
                data=csv,
                file_name=f"aqi_export_{start_date}_{end_date}.csv",
                mime="text/csv",
            )


# Tab 3: Sobre el proyecto

with tab_about:
    st.subheader("Sobre el proyecto")

    # ── Resumen ejecutivo ────────────────────────────────────────────────
    st.markdown(
        """
        Sistema de clasificación de calidad del aire **multinacional** basado
        en Machine Learning supervisado, con pipeline de reentrenamiento
        completamente automatizado en la nube. Trabajo final del curso de
        **Cloud Computing** — **USIL**.
        """
    )

    # ── Tarjetas de resumen ──────────────────────────────────────────────
    info = call_model_info()
    if info:
        st.markdown("### Modelo en producción")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tipo", info["model_type"].replace("Classifier", ""))
        c2.metric("Accuracy (test)", f"{info['metrics']['accuracy_test']:.2%}")
        c3.metric(
            "F1 macro",
            f"{info['metrics'].get('f1_macro', 0):.2%}" if "f1_macro" in info.get("metrics", {}) else "—",
        )
        c4.metric("Países", len(info.get("countries", [])) or "—")
    else:
        st.info("No se pudo conectar al backend para mostrar la metadata del modelo.")

    # ── Pestañas para no abrumar ─────────────────────────────────────────
    sub_data, sub_model, sub_stack, sub_journey, sub_meta = st.tabs(
        ["Datos", "Modelo", "Stack tecnológico", "Iteraciones del proyecto", "Metadata"]
    )

    with sub_data:
        st.markdown(
            """
            #### Fuente de datos: OpenAQ

            [OpenAQ](https://openaq.org) es una ONG global que agrega mediciones
            de calidad del aire de **gobiernos, agencias ambientales y sensores
            ciudadanos**. Sus datos están publicados como _AWS Open Data_ —
            accesibles vía:

            - **REST API** (catálogo de estaciones, metadata)
            - **S3 bucket público** (`openaq-data-archive`) con mediciones
              históricas en CSV particionado.

            #### Países cubiertos

            Lista curada por **alta concentración de PM2.5** y buena cobertura
            de estaciones:

            | Asia del Sur | Sudeste Asiático | Asia Central | Latam |
            |---|---|---|---|
            | 🇧🇩 Bangladesh | 🇹🇭 Tailandia | 🇰🇿 Kazajistán | 🇲🇽 México |
            | 🇮🇳 India | 🇻🇳 Vietnam | 🇲🇳 Mongolia | |
            | 🇵🇰 Pakistán | 🇮🇩 Indonesia | | |
            | 🇳🇵 Nepal | | | |

            #### Volumen

            - **~1.7M mediciones** procesadas (ventana móvil de 12 meses).
            - **~400 estaciones** únicas activas.
            - Datos actualizados en cada reentrenamiento semanal.

            #### Limpieza aplicada

            - Filtros: requiere PM2.5 disponible; drop de columnas con >70% NaN.
            - Imputación: mediana global para variables meteorológicas faltantes.
            - Clipping: valores fuera de rango físico razonable (sensores rotos).
            """
        )

    with sub_model:
        st.markdown(
            """
            #### Algoritmo

            **Random Forest Classifier** con `class_weight="balanced"` para
            compensar el desbalance natural entre clases AQI.

            **Hiperparámetros:**
            - `n_estimators = 100` árboles
            - `max_depth = 20`
            - `min_samples_leaf = 50`

            #### Variable objetivo: 5 clases EPA

            | Clase | PM2.5 (µg/m³) | Etiqueta |
            |:----:|:---:|---|
            | 0 | 0 – 12 | 🟢 Buena |
            | 1 | 12.1 – 35.4 | 🟡 Moderada |
            | 2 | 35.5 – 55.4 | 🟠 Dañina para grupos sensibles |
            | 3 | 55.5 – 150.4 | 🔴 Dañina |
            | 4 | 150.5+ | 🟣 Muy dañina |

            #### Gate de calidad

            El pipeline solo publica el modelo a producción si supera un
            umbral mínimo de accuracy (`MIN_ACCURACY = 0.70`). Esto previene
            que un reentrenamiento con datos degradados rompa el servicio.

            #### Limitaciones honestas

            - El modelo **no predice** PM2.5 numérico; clasifica en bandas EPA.
            - El accuracy varía por país: mejor en regiones con más estaciones
              (India, Pakistán) que en las de menor cobertura (Mongolia).
            - No usa contaminantes secundarios (O₃, NO₂) porque el >70% de
              estaciones no los reportan.
            """
        )

    with sub_stack:
        st.markdown(
            """
            #### Capa de datos

            - **OpenAQ S3** (`openaq-data-archive`) → data lake público.
            - **AWS Glue Data Catalog** → metadata + schema externo.
            - **AWS Athena** → query SQL serverless sobre el data lake con
              partition projection (escanea solo lo necesario).
            - **AWS S3** → buckets propios para data preparada y artefactos
              del modelo.
            - **Supabase Postgres** → caché de visualización para el dashboard.

            #### Capa de ML

            - **Databricks Free Edition** (serverless) → reentrenamiento
              programado del Random Forest.
            - **scikit-learn 1.2.2** → algoritmo + serialización con `joblib`.

            #### Capa de servicio

            - **AWS EC2** + **Docker** → contenedor con la API FastAPI.
            - **AWS IAM** → roles de servicio con principio de _least privilege_.
            - **FastAPI** + **Uvicorn** → endpoint REST `/predict` y `/model-info`.
            - **Streamlit Community Cloud** → frontend interactivo (esta app).

            #### Capa de automatización

            - **Databricks Workflows** → job semanal que dispara el pipeline
              Athena → entrenamiento → S3.
            """
        )

    with sub_journey:
        st.markdown(
            """
            El proyecto evolucionó a través de varias iteraciones, cada una
            resolviendo limitaciones de la anterior:

            ##### Fase 1 – MVP local
            Modelo entrenado en notebook Jupyter sobre un CSV descargado de
            Kaggle (Almaty, ~517K filas, 1 país). Bundleado dentro de un Docker
            image y desplegado en Render.

            ##### Fase 2 – Migración a AWS
            API movida a una **EC2** con **IAM role** para acceder al modelo
            en **S3** sin credenciales hardcodeadas. Modelo desacoplado de la
            imagen Docker: la API descarga `rf_aqi.pkl` al iniciar.

            ##### Fase 3 – Multipaís
            Reemplazo del CSV de Almaty por **OpenAQ multinacional**. Pipeline
            de descarga vía API + S3 paralelo (10 países, 3 meses, ~186K filas).
            Frontend rediseñado con dropdown de país y dashboard analítico.

            ##### Fase 4 – Data lake con Athena + Glue
            Eliminado el bottleneck de descarga paralela. Tabla externa en
            **Glue Data Catalog** apuntando al S3 público de OpenAQ. Queries
            SQL desde **Athena** con _partition projection_ — la ventana
            crece a 12 meses (~1.7M filas) sin penalización de tiempo.

            ##### Fase 5 – Entrenamiento automatizado
            Reentrenamiento programado en **Databricks Free Edition**. El
            notebook es autónomo: consulta Athena, entrena, valida gate de
            calidad y publica a S3. Sin intervención manual.
            """
        )

    with sub_meta:
        if info:
            st.markdown("#### Metadata completa del modelo en producción")
            st.json(info)
        else:
            st.info("Conecta el backend para ver la metadata.")

# Footer

st.markdown("---")
st.caption(
    f"Backend: {'conectado' if call_model_info() else 'no disponible'} · "
    f"Supabase: {'configurado' if SUPABASE_URL else 'no configurado'}"
)
