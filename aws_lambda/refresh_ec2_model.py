"""Lambda que refresca el contenedor Docker de la API en la EC2 cuando
sube un nuevo modelo a S3.

Trigger: s3:ObjectCreated:Put sobre s3://aqi-almaty-models-ee/rf_aqi.pkl

Flujo:
1. Databricks (job semanal) entrena el modelo y sube rf_aqi.pkl a S3.
2. S3 emite un evento s3:ObjectCreated:Put.
3. Esta Lambda recibe el evento.
4. Lambda llama a SSM SendCommand sobre la EC2.
5. EC2 ejecuta: docker stop → docker rm → rm cache → docker run.
6. El contenedor nuevo descarga el modelo recién subido de S3.

Variables de entorno (configuradas en la consola Lambda):
- EC2_INSTANCE_ID    ID de la instancia (i-xxxxxxxx)
- MODEL_BUCKET       Default: aqi-almaty-models-ee
- AWS_REGION         Default: us-east-1
- DOCKER_IMAGE       Default: aqi-api

IAM execution role de la Lambda necesita:
- AWSLambdaBasicExecutionRole (logs CloudWatch)
- ssm:SendCommand sobre la EC2 específica

EC2 instance role necesita:
- AmazonSSMManagedInstanceCore (para que el SSM Agent reciba comandos)
- AmazonS3ReadOnly sobre el model bucket (ya lo tenía)
"""
import json
import os

import boto3


def lambda_handler(event, context):
    """Recibe el evento de S3 y dispara el restart del contenedor en EC2."""
    instance_id = os.environ["EC2_INSTANCE_ID"]
    model_bucket = os.environ.get("MODEL_BUCKET", "aqi-almaty-models-ee")
    aws_region = os.environ.get("AWS_REGION", "us-east-1")
    docker_image = os.environ.get("DOCKER_IMAGE", "aqi-api")

    # Log básico de qué disparó la Lambda
    s3_records = event.get("Records", [])
    triggers = [
        f"{r.get('s3', {}).get('bucket', {}).get('name')}/"
        f"{r.get('s3', {}).get('object', {}).get('key')}"
        for r in s3_records
    ]
    print(f"Triggered by S3 events: {triggers}")
    print(f"Target instance: {instance_id} in {aws_region}")

    # Los comandos shell que correrán en la EC2.
    # set -e detiene si algo falla. Los `|| true` toleran que el contenedor
    # ya esté apagado (primera corrida no hay nada que matar).
    commands = [
        "set -e",
        "cd /home/ec2-user || cd /root",
        "echo '[refresh] Stopping current container...'",
        "docker stop aqi-api || true",
        "docker rm aqi-api || true",
        "echo '[refresh] Clearing model cache...'",
        "rm -rf /tmp/aqi-model",
        "echo '[refresh] Starting new container...'",
        (
            f"docker run -d --name aqi-api -p 8000:8000 "
            f"-e S3_BUCKET={model_bucket} "
            f"-e AWS_REGION={aws_region} "
            f"--restart unless-stopped "
            f"{docker_image}"
        ),
        "echo '[refresh] Done. Container status:'",
        "docker ps --filter name=aqi-api",
    ]

    ssm = boto3.client("ssm", region_name=aws_region)
    response = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": commands},
        TimeoutSeconds=300,
        Comment=f"Refresh aqi-api after S3 update from {triggers}",
    )

    command_id = response["Command"]["CommandId"]
    print(f"SSM command sent: {command_id}")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "command_id": command_id,
            "instance_id": instance_id,
            "triggered_by": triggers,
        }),
    }
