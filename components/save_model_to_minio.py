from kfp import dsl
from kfp.dsl import Input, Artifact, Model, Metrics

BASE_IMAGE = "image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/s2i-generic-data-science-notebook:2025.2"

@dsl.component(base_image=BASE_IMAGE)
def save_model_to_minio(
    s3_endpoint: str,
    models_bucket: str,
    model_prefix: str,   # e.g. "churn/baseline/<run_id>"
    trained_model: Input[Model],
    preprocessor: Input[Artifact],
    metrics: Input[Metrics],
):
    import os
    import json
    import boto3

    # Connect to MinIO/S3 using injected secret
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )

    def upload(local_path: str, key: str):
        with open(local_path, "rb") as f:
            s3.put_object(Bucket=models_bucket, Key=key, Body=f)

    # Upload model + preprocessor binaries
    upload(trained_model.path, f"{model_prefix}/model.joblib")
    upload(preprocessor.path, f"{model_prefix}/preprocessor.joblib")

    # Metrics is stored as a metadata file by KFP; upload a clean JSON as well
    # We'll read any existing metrics content if present, otherwise upload an empty dict.
    metrics_payload = {}
    try:
        # KFP Metrics typically stores a JSON at metrics.path
        with open(metrics.path, "r", encoding="utf-8") as f:
            metrics_payload = json.load(f)
    except Exception:
        pass

    metrics_key = f"{model_prefix}/metrics.json"
    s3.put_object(
        Bucket=models_bucket,
        Key=metrics_key,
        Body=json.dumps(metrics_payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    print(f"Uploaded: s3://{models_bucket}/{model_prefix}/model.joblib")
    print(f"Uploaded: s3://{models_bucket}/{model_prefix}/preprocessor.joblib")
    print(f"Uploaded: s3://{models_bucket}/{metrics_key}")
