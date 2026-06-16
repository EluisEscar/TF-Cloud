# Sistema de Clasificación de Calidad del Aire — Cloud Computing (USIL)

Sistema end-to-end de Machine Learning para clasificar la calidad del aire en
**10 países** con alta concentración de PM2.5, con pipeline de reentrenamiento
automatizado en la nube.

Trabajo final del curso de **Cloud Computing — USIL**.

---

## Arquitectura final

```
                         ┌──────────────────────────────┐
                         │   OpenAQ Open Data Platform  │
                         │   s3://openaq-data-archive   │  (público, AWS Open Data)
                         └──────────────┬───────────────┘
                                        │ external table
                         ┌──────────────▼───────────────┐
                         │   AWS Glue Data Catalog      │
                         │   openaq_aqi.records         │  (schema + partition projection)
                         └──────────────┬───────────────┘
                                        │ SQL queries
                         ┌──────────────▼───────────────┐
                         │   AWS Athena (serverless)    │
                         └──────────────┬───────────────┘
                                        │ resultados
                         ┌──────────────▼───────────────┐
                         │   Databricks Job (semanal)   │  cron 0 2 * * 0 (UTC)
                         │   - lee Athena               │
                         │   - entrena Random Forest    │
                         │   - gate de calidad          │
                         │   - publica a S3             │
                         └──────────────┬───────────────┘
                                        │ s3:ObjectCreated
                         ┌──────────────▼───────────────┐
                         │   S3: aqi-almaty-models-ee   │
                         │   - rf_aqi.pkl               │
                         │   - features.json            │
                         └──────────────┬───────────────┘
                                        │ trigger
                         ┌──────────────▼───────────────┐
                         │   AWS Lambda                 │
                         │   refresh-aqi-model          │
                         └──────────────┬───────────────┘
                                        │ SSM SendCommand
                         ┌──────────────▼───────────────┐
                         │   EC2 t3.micro + Docker      │
                         │   FastAPI (puerto 8000)      │  ← descarga modelo nuevo de S3
                         └──────────────┬───────────────┘
                                        │ HTTPS
        ┌───────────────────────────────┴──────────────────────────────┐
        │                                                              │
┌───────▼────────────────┐                              ┌──────────────▼──────────────┐
│  Streamlit Cloud       │                              │  Supabase Postgres          │
│  Predicción + Dashboard│                              │  caché de datos históricos  │
│  (3 tabs)              │                              │  (para mapa + time series)  │
└────────────────────────┘                              └─────────────────────────────┘
```

**Demo en vivo:**
- Frontend: ver Streamlit Cloud URL
- API Swagger: `http://<IP_EC2>:8000/docs` (HTTP, sin TLS — academic project)

---

## Las 5 fases del proyecto

El proyecto evolucionó iterativamente. Cada fase resuelve una limitación
de la anterior.

| Fase | Foco | Tecnologías añadidas |
|---|---|---|
| **1** | MVP local con CSV de Almaty | pandas, scikit-learn, Jupyter |
| **2** | Deploy básico | FastAPI, Docker, Render, Streamlit Cloud, Supabase |
| **3** | Migración a AWS | EC2, S3, IAM roles |
| **4** | Multipaís + data lake | OpenAQ S3, **Athena, Glue** |
| **5** | Automatización completa | **Databricks Jobs**, **Lambda**, **SSM** |

Detalles de cada fase: ver tab "Sobre el proyecto" en la app Streamlit.

---

## Modelo en producción

- **Algoritmo:** `RandomForestClassifier` (sklearn 1.6.1)
- **Features (12):** PM1, PM10, humedad, temperatura, um003, lat, lon, hour,
  day, month, year, dayofweek, country_id
- **Target:** AQI class (0-4) según breakpoints EPA para PM2.5
- **Dataset:** ~1.7M mediciones de 10 países OpenAQ, ventana móvil de 12 meses
- **Accuracy (test):** ~0.84 | **F1 macro:** ~0.85
- **Reentreno:** automático cada domingo 02:00 UTC

---

## Estructura del repo

```
TF-Cloud/
├── api/                    # FastAPI app (Docker, deploy en EC2)
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── aws_lambda/             # Lambda que refresca la EC2 al subir modelo
│   └── refresh_ec2_model.py
├── databricks/             # Notebook autónomo para training semanal
│   └── train_aqi.py
├── frontend/               # App Streamlit (3 tabs)
│   └── app.py
├── ml_pipeline/            # Scripts ETL + training launcher
│   ├── explore_openaq.py       # Exploración inicial
│   ├── prepare_data.py         # ETL vía API + S3 (legacy)
│   ├── prepare_data_athena.py  # ETL vía Athena (recomendado)
│   ├── athena_setup.py         # Bootstrap Glue + tabla externa
│   └── upload_to_supabase.py   # Sincroniza Parquet → Supabase
├── tests/                  # Pruebas de rendimiento (load, stress, availability)
│   ├── load_test.py
│   └── README.md
├── src/legacy/             # Código de la Fase 1 (Almaty CSV)
├── notebooks/              # EDA y modelado original (Fase 1)
├── data/                   # (gitignored) Parquet local
├── models/                 # (gitignored) .pkl bundleado para fallback
├── requirements.txt        # Dev dependencies
└── render.yaml             # Deploy config de Render (legacy)
```

---

## Cómo correr cada pieza

### 1. Frontend local

```bash
pip install -r requirements.txt
streamlit run frontend/app.py
```

Requiere `.env` con `API_URL`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`. Ver
`.env.example`.

### 2. API local

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload
```

Por defecto sirve el `.pkl` bundleado en `models/`. Para descargar de S3 al
arrancar, define `S3_BUCKET=aqi-almaty-models-ee` y `AWS_REGION=us-east-1`.

### 3. ETL + Training (manual)

```bash
# Primero (una vez): bootstrap de Athena
python ml_pipeline/athena_setup.py

# Cada vez que quieras re-procesar datos
python ml_pipeline/prepare_data_athena.py --months 12

# Subir el Parquet a S3 para que Databricks lo consuma
python ml_pipeline/prepare_data_athena.py --months 12 --upload-s3

# Sincronizar Supabase con los datos nuevos (para el dashboard)
python ml_pipeline/upload_to_supabase.py --truncate
```

### 4. Reentrenamiento programado

El notebook `databricks/train_aqi.py` es **autónomo** — consulta Athena, entrena,
publica a S3 sin depender de tu PC. Configura como Job semanal:

- Schedule: `0 2 * * 0` (domingos 02:00 UTC)
- Compute: serverless
- Secrets en scope `aqi`: `openaq_api_key`, `aws_access_key_id`, `aws_secret_access_key`

### 5. Pruebas de rendimiento

```bash
export API_URL=http://<IP_EC2>:8000

# Load test
python tests/load_test.py --mode load --concurrent 10 --duration 60

# Stress test
python tests/load_test.py --mode stress --max-concurrent 200 --step 10

# Availability test
python tests/load_test.py --mode availability --duration 600 --interval 5
```

Ver `tests/README.md` para detalles sobre interpretación.

---

## Pipeline automatizado completo

Una vez configurado (Fases 1-5), el sistema funciona solo:

```
Domingo 02:00 UTC
    ↓
Databricks Job arranca compute serverless
    ↓
Notebook: API OpenAQ + Athena + train RF
    ↓
Gate de calidad (accuracy_test ≥ 0.70)
    ↓
Sube rf_aqi.pkl + features.json a S3
    ↓
S3 event → Lambda invocada
    ↓
Lambda → SSM SendCommand sobre EC2
    ↓
EC2: docker restart aqi-api (con nuevo modelo)
    ↓
Streamlit predice con el modelo fresco
```

**Cero intervención manual semanal.** El sistema se mantiene actualizado solo.

---

## Stack tecnológico

### Datos
- **OpenAQ Open Data Platform** — agregador global de mediciones de aire.
- **AWS S3** — data lake y storage de artefactos del modelo.
- **AWS Glue Data Catalog** — metadata + schema externo.
- **AWS Athena** — SQL serverless sobre el data lake.
- **Supabase Postgres** — caché de visualización del frontend.

### ML
- **scikit-learn 1.6.1** — Random Forest + serialización joblib.
- **Databricks Free Edition (serverless)** — reentrenamiento semanal.

### Servicio
- **FastAPI + Uvicorn** — API REST.
- **Docker** — contenedor de la API.
- **AWS EC2 (t3.micro)** — host del Docker.
- **AWS IAM** — roles con principio de least privilege.
- **Streamlit Community Cloud** — frontend.

### Automatización
- **Databricks Workflows** — scheduler del retraining.
- **AWS Lambda + EventBridge** — trigger por evento de S3.
- **AWS Systems Manager (SSM)** — ejecución remota en EC2.

---

## Limitaciones conocidas

- **HTTP sin TLS** en la API (academic project; no apto para datos sensibles).
- **Free Tier de AWS expira en 6 meses** ($200 de crédito).
- **`t3.micro` satura a ~80-150 req/s** (ver pruebas en `tests/`).
- **Modelo solo PM2.5**: no usa contaminantes secundarios (O₃, NO₂) porque >70 %
  de estaciones no los reportan.
- **Datos OpenAQ sin SLA**: pueden faltar mediciones recientes.

---

## Autoría

Estephany Camposano · Curso Cloud Computing · USIL.
