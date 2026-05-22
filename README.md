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

## Fases del proyecto

- [x] **Fase 1:** Setup del entorno y estructura
- [x] **Fase 2:** EDA y limpieza del dataset
- [x] **Fase 3:** Entrenamiento del Random Forest
- [x] **Fase 4:** Carga de datos históricos a Supabase
- [x] **Fase 5:** API REST con FastAPI + Docker
- [ ] **Fase 6:** Frontend con Streamlit
- [ ] **Fase 7:** Deploy en Render + Streamlit Community Cloud

## Autoría

Trabajo final del curso de Cloud Computing — USIL.
