"""Lanza el training job de SageMaker desde tu PC.

Lee credenciales/region del .env (las del IAM user aqi-local-dev), crea
el estimador SKLearn apuntando al data bucket, y dispara el job. SageMaker
provisiona una ml.m5.large efímera (~$0.10/h), corre ml_pipeline/train.py
dentro de su container, sube los artefactos a S3 y se autoapaga.

Uso:
    python ml_pipeline/launch_training.py

Variables de entorno (desde .env):
    SAGEMAKER_ROLE_ARN   ARN del role aqi-sagemaker-role
    DATA_BUCKET          default: aqi-almaty-data-ee
    MODEL_BUCKET         default: aqi-almaty-models-ee
    AWS_REGION           default: us-east-1
    SM_INSTANCE_TYPE     default: ml.m5.large  (~$0.10/h)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker.sklearn.estimator import SKLearn

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("launch")

# ─── Config ────────────────────────────────────────────────────────────────
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "")
DATA_BUCKET = os.getenv("DATA_BUCKET", "aqi-almaty-data-ee")
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "aqi-almaty-models-ee")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
INSTANCE_TYPE = os.getenv("SM_INSTANCE_TYPE", "ml.m5.large")

if not ROLE_ARN:
    raise SystemExit(
        "Falta SAGEMAKER_ROLE_ARN en tu .env. Pega el ARN del role aqi-sagemaker-role."
    )

DATA_S3_URI = f"s3://{DATA_BUCKET}/processed/"
OUTPUT_S3_URI = f"s3://{MODEL_BUCKET}/sagemaker/"

# Nombre único del job (necesario, SageMaker no permite duplicados)
JOB_NAME = "aqi-rf-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

log.info("Configuración del training job:")
log.info("  role          = %s", ROLE_ARN)
log.info("  data input    = %s", DATA_S3_URI)
log.info("  model bucket  = %s", MODEL_BUCKET)
log.info("  output (tar)  = %s", OUTPUT_S3_URI)
log.info("  instance      = %s", INSTANCE_TYPE)
log.info("  job name      = %s", JOB_NAME)
log.info("  region        = %s", AWS_REGION)

# ─── Sesión boto3/sagemaker con región explícita ───────────────────────────
# El SDK falla con "Must setup local AWS configuration with a region..." si
# no encuentra la región en ~/.aws/config o en AWS_DEFAULT_REGION. Inyectamos
# la región leída del .env directamente en la sesión, así no depende del
# entorno del usuario.
boto_session = boto3.Session(region_name=AWS_REGION)
sagemaker_session = sagemaker.Session(boto_session=boto_session)

# ─── Estimator ─────────────────────────────────────────────────────────────
# framework_version 1.2-1 corresponde a scikit-learn 1.2.1 — viene con
# pandas + joblib + boto3 preinstalados. Si train.py necesita más, los
# añadimos vía ml_pipeline/requirements.txt (se instala automáticamente).
sklearn_estimator = SKLearn(
    entry_point="train.py",
    source_dir="ml_pipeline",
    role=ROLE_ARN,
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    output_path=OUTPUT_S3_URI,
    base_job_name="aqi-rf",
    sagemaker_session=sagemaker_session,
    hyperparameters={
        "model-bucket": MODEL_BUCKET,
        "n-estimators": 100,
        "max-depth": 20,
        "min-samples-leaf": 50,
        "min-accuracy": 0.65,
    },
    environment={
        "AWS_DEFAULT_REGION": AWS_REGION,
    },
)

# ─── Lanzar ────────────────────────────────────────────────────────────────
log.info("Lanzando training job (puede tardar 3-8 min en total) ...")
sklearn_estimator.fit({"train": DATA_S3_URI}, job_name=JOB_NAME, wait=True)

log.info("✅ Training completado.")
log.info("Model tarball: %s/output/model.tar.gz", sklearn_estimator.output_path)
log.info("Modelo desempaquetado en: s3://%s/rf_aqi.pkl + features.json", MODEL_BUCKET)
log.info("CloudWatch logs: %s", sklearn_estimator.latest_training_job.describe()["TrainingJobName"])
