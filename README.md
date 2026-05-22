# Sistema de Clasificación de Calidad del Aire — Cloud Computing (USIL)

MVP end-to-end de un sistema de Machine Learning para clasificar la calidad del aire en Almaty (Kazajistán) usando PM2.5 como referencia, desplegado en la nube.

## Contexto académico

- **Universidad:** USIL — Universidad San Ignacio de Loyola, Lima
- **Curso:** Cloud Computing
- **Dataset:** [Almaty Air Quality History](https://www.kaggle.com/datasets/fichka/almaty-air-quality-history) (Kaggle, +500K registros, 2020-2026, fuente original OpenAQ)
- **Objetivo:** Clasificación supervisada del nivel de calidad del aire usando los breakpoints EPA para PM2.5

## Arquitectura

```
Dataset CSV (Kaggle)
  ↓
EDA + limpieza local (Jupyter + pandas)
  ↓
Random Forest entrenado → models/rf_aqi.pkl
  ↓
Datos históricos limpios → Supabase (PostgreSQL)
  ↓
FastAPI /predict ─────────────── Streamlit (frontend)
        │                              │
        │                              ↓
        │                        Consulta Supabase
        │                        para mapas e históricos
        ↓                              ↓
Docker + Render               Streamlit Community Cloud
```

## Stack

| Capa | Tecnología |
|------|------------|
| Lenguaje | Python 3.11+ |
| ML | scikit-learn, pandas, numpy |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Base de datos | Supabase (PostgreSQL gestionado) |
| Contenedor | Docker |
| Deploy API | Render |
| Deploy Frontend | Streamlit Community Cloud |
| Control de versiones | GitHub |

## Variable objetivo

Clasificación de PM2.5 (µg/m³) en 5 clases según breakpoints EPA:

| Clase | Rango | Etiqueta |
|------:|-------|----------|
| 0 | 0 – 12 | Buena |
| 1 | 12.1 – 35.4 | Moderada |
| 2 | 35.5 – 55.4 | Dañina para grupos sensibles |
| 3 | 55.5 – 150.4 | Dañina |
| 4 | 150.5+ | Muy dañina / Peligrosa |

## Estructura del proyecto

```
TF-Cloud/
├── data/                    # CSVs (en .gitignore)
├── notebooks/
│   └── 01_eda.ipynb         # EDA, limpieza y entrenamiento
├── models/
│   └── rf_aqi.pkl           # Modelo entrenado (en .gitignore)
├── api/
│   ├── main.py              # FastAPI
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py               # Streamlit
│   └── requirements.txt
├── src/
│   ├── preprocessing.py     # Funciones de limpieza
│   └── upload_to_supabase.py
├── .gitignore
├── requirements.txt         # Dependencias de desarrollo / notebooks
└── README.md
```

## Setup local

### 1. Crear y activar entorno virtual

```bash
python3.11 -m venv venv
source venv/bin/activate          # Linux / macOS
# .\venv\Scripts\activate         # Windows PowerShell
```

### 2. Instalar dependencias de desarrollo

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Descargar el dataset de Kaggle

Necesitas tener `~/.kaggle/kaggle.json` configurado (descargable desde tu cuenta de Kaggle → Account → Create New API Token).

```bash
mkdir -p data
kaggle datasets download -d fichka/almaty-air-quality-history -p data --unzip
```

Tras descomprimir, verás los CSVs dentro de `data/`. El notebook de la Fase 2 ajustará la ruta exacta.

### 4. Registrar el kernel del venv en Jupyter (opcional)

```bash
python -m ipykernel install --user --name=air-quality-cloud --display-name "Python (air-quality-cloud)"
```

## Configurar Supabase

1. Crea una cuenta gratuita en [supabase.com](https://supabase.com).
2. **New project** → ponle un nombre (p. ej. `air-quality-almaty`), elige una contraseña segura para la DB y la región más cercana (p. ej. `us-east-1`). Guarda la contraseña.
3. Espera ~2 min a que el proyecto se aprovisione.
4. Toma las credenciales:
   - **Connection string (DATABASE_URL):** Settings → Database → *Connection string* → URI. Usa el modo **Transaction pooler** (puerto 6543). Reemplaza `[YOUR-PASSWORD]` por la contraseña del paso 2.
   - **URL y anon key:** Settings → API → *Project URL* y *anon public*.
5. Copia `.env.example` a `.env` y rellena los valores:
   ```bash
   cp .env.example .env     # macOS/Linux
   copy .env.example .env   # Windows
   ```
6. Crea la tabla e índices y carga los datos históricos:
   ```bash
   # Solo la tabla, sin insertar (para verificar conexión primero):
   python src/upload_to_supabase.py --schema-only

   # Prueba con 10K filas para validar:
   python src/upload_to_supabase.py --truncate --limit 10000

   # Carga completa (~517K filas, puede tardar varios minutos):
   python src/upload_to_supabase.py --truncate
   ```
7. Verifica en Supabase Studio → Table Editor que existe la tabla `air_quality` con los datos.

## API local (FastAPI)

La API REST sirve predicciones del modelo entrenado.

### Correr localmente sin Docker

```bash
# Desde la raíz del proyecto, con el venv activado
pip install -r api/requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Abre [http://localhost:8000/docs](http://localhost:8000/docs) para la UI Swagger interactiva.

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Healthcheck |
| GET | `/model-info` | Metadata del modelo (features, clases, métricas) |
| POST | `/predict` | Clasifica una observación → clase + label + probabilidades |

### Probar `/predict` con curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pm10":35,"relativehumidity":70,"temperature":-5,"um003":2500,"hour":12,"day":15,"month":1,"year":2025,"dayofweek":2,"lat":43.25,"lon":76.93}'
```

### Correr con Docker

```bash
# Build (desde la raíz del proyecto, NO desde api/)
docker build -f api/Dockerfile -t aqi-api .

# Run
docker run --rm -p 8000:8000 aqi-api
```

## Frontend (Streamlit)

App con 3 tabs: **Predicción en tiempo real**, **Histórico** (mapa, evolución, distribución desde Supabase) y **Sobre el proyecto**.

### Correr localmente

```bash
# Asegúrate de tener la API corriendo en otra terminal (uvicorn api.main:app)
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

Se abrirá en [http://localhost:8501](http://localhost:8501).

El frontend lee la config de **dos sitios** (en este orden):
1. `frontend/.streamlit/secrets.toml` (si existe)
2. Variables de entorno / `.env` raíz

Para desarrollo local, basta con el `.env` raíz que ya tienes (con `API_URL`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`).

Para Streamlit Community Cloud, configurarás los secrets en su UI (ver Fase 7).

## Deploy a la nube

Arquitectura objetivo:

```
GitHub (este repo)
   │
   ├─→ Render          (Web Service · Docker · api/Dockerfile)  → API pública
   └─→ Streamlit Cloud (frontend/app.py)                        → Frontend público
                                │
                                ├─ llama /predict en Render
                                └─ consulta Supabase para histórico
```

### 1. Subir el repo a GitHub

Si todavía no tienes el repo en GitHub:

```bash
# Crea un repo vacío en github.com/<tu-usuario>/TF-Cloud (sin README ni .gitignore)

git remote add origin https://github.com/<tu-usuario>/TF-Cloud.git
git branch -M main
git push -u origin main
```

Si ya tienes remoto y solo subiste una rama de trabajo, abre PR → mergea a `main`. Render y Streamlit Cloud apuntarán a `main` por defecto.

**Antes de pushear, verifica que NO subes secretos:**

```bash
git ls-files | grep -E '\.env$|secrets\.toml$|kaggle\.json$'
# No debe devolver nada. Si devuelve algo, sácalo del index antes del push.
```

### 2. Deploy de la API en Render

El `api/Dockerfile` está preparado para Render: lee `$PORT`, expone `/` como healthcheck y copia el modelo dentro de la imagen.

**Opción A — Blueprint (recomendado, usa `render.yaml`):**

1. Entra a [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**.
2. Conecta tu cuenta de GitHub y selecciona el repo.
3. Render detectará automáticamente `render.yaml` y propondrá crear el servicio `aqi-api`.
4. Confirma y haz **Apply**.

**Opción B — Web Service manual (sin Blueprint):**

1. **New** → **Web Service** → conecta el repo.
2. Configura:
   | Campo | Valor |
   |---|---|
   | Name | `aqi-api` (o el que prefieras) |
   | Region | la más cercana (p. ej. `Oregon`) |
   | Branch | `main` |
   | Runtime | **Docker** |
   | Dockerfile Path | `api/Dockerfile` |
   | Docker Build Context Directory | `.` (raíz del repo) |
   | Instance Type | **Free** |
   | Health Check Path | `/` |
3. **Create Web Service**. El primer build tarda ~5 min (instala scikit-learn).
4. Cuando termine, copia la URL pública. Tendrá la forma:
   ```
   https://aqi-api-xxxx.onrender.com
   ```
5. Verifica que la API responde:
   ```bash
   curl https://aqi-api-xxxx.onrender.com/
   # {"status":"ok","model_loaded":true}

   curl https://aqi-api-xxxx.onrender.com/docs
   # Devuelve el HTML de Swagger UI
   ```

**Nota sobre el plan Free de Render:** la instancia se duerme tras ~15 min de inactividad. La primera petición tras dormirse tarda ~30-60 s (cold start). El frontend tiene timeout de 15 s en `requests.post`, así que la primera predicción tras un periodo de inactividad puede fallar — vuelve a intentar.

### 3. Deploy del frontend en Streamlit Community Cloud

1. Entra a [share.streamlit.io](https://share.streamlit.io) → **Sign in with GitHub**.
2. **New app** → **From existing repo**.
3. Configura:
   | Campo | Valor |
   |---|---|
   | Repository | `<tu-usuario>/TF-Cloud` |
   | Branch | `main` |
   | Main file path | `frontend/app.py` |
   | App URL | el subdominio que prefieras (p. ej. `aqi-almaty`) |
4. Antes de deployar, abre **Advanced settings → Secrets** y pega:
   ```toml
   API_URL = "https://aqi-api-xxxx.onrender.com"
   SUPABASE_URL = "https://<PROJECT_REF>.supabase.co"
   SUPABASE_ANON_KEY = "eyJhbGciOi..."
   ```
   Streamlit Cloud guarda esto encriptado y lo expone como `st.secrets[...]`. El código del frontend ya está preparado para leer de ahí (`get_setting` en `frontend/app.py`).
5. **Deploy**. El primer build tarda ~3 min. Cuando termine, la URL final es:
   ```
   https://aqi-almaty.streamlit.app
   ```
6. Abre la app y prueba las 3 tabs:
   - **Predicción:** debe llamar al endpoint de Render y devolver clase + probabilidades.
   - **Histórico:** debe consultar Supabase y pintar el mapa + serie temporal.
   - **Sobre el proyecto:** debe mostrar la metadata del modelo desde `/model-info`.

### 4. (Opcional) Restringir CORS al dominio de Streamlit

En `api/main.py` el CORS está abierto (`allow_origins=["*"]`) para no bloquear el MVP. Cuando tengas la URL final del frontend, cámbialo a:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aqi-almaty.streamlit.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Push a `main` → Render redeployará automáticamente (`autoDeploy: true` en `render.yaml`).

### 5. Checklist final

- [ ] La URL de Render responde `200` en `/` y muestra `/docs` en Swagger.
- [ ] La URL de Streamlit carga las 3 tabs sin errores.
- [ ] Una predicción de prueba en el frontend devuelve clase + probabilidades.
- [ ] La tab "Histórico" muestra el mapa con estaciones reales (no vacío).
- [ ] La tab "Sobre el proyecto" muestra accuracy y nº de features (vienen de `/model-info`).

## Fases del proyecto

- [x] **Fase 1:** Setup del entorno y estructura
- [x] **Fase 2:** EDA y limpieza del dataset
- [x] **Fase 3:** Entrenamiento del Random Forest
- [x] **Fase 4:** Carga de datos históricos a Supabase
- [x] **Fase 5:** API REST con FastAPI + Docker
- [x] **Fase 6:** Frontend con Streamlit
- [x] **Fase 7:** Deploy en Render + Streamlit Community Cloud

## Autoría

Trabajo final del curso de Cloud Computing — USIL.
