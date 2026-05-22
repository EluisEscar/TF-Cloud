FROM python:3.11-slim

# Mejora arranque y logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencias del sistema mínimas para numpy/scikit-learn
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# Instala deps primero para aprovechar cache de capas
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install -r /app/api/requirements.txt

# Copia código de la API y modelo entrenado
COPY api/ /app/api/
COPY models/ /app/models/

# Render inyecta $PORT; localmente usamos 8000 por defecto.
ENV PORT=8000
EXPOSE 8000

# Healthcheck sencillo (opcional, ayuda al orquestador)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen(f'http://localhost:{__import__(\"os\").environ.get(\"PORT\",\"8000\")}/').status == 200 else 1)" || exit 1

# Arranque: uvicorn lee $PORT (Render lo setea)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
