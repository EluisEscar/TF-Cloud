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

## Fases del proyecto

- [x] **Fase 1:** Setup del entorno y estructura
- [ ] **Fase 2:** EDA y limpieza del dataset
- [ ] **Fase 3:** Entrenamiento del Random Forest
- [ ] **Fase 4:** Carga de datos históricos a Supabase
- [ ] **Fase 5:** API REST con FastAPI + Docker
- [ ] **Fase 6:** Frontend con Streamlit
- [ ] **Fase 7:** Deploy en Render + Streamlit Community Cloud

## Autoría

Trabajo final del curso de Cloud Computing — USIL.
