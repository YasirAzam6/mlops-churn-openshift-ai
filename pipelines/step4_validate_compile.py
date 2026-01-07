from kfp import dsl
from kfp.compiler import Compiler
from kfp import kubernetes

from components.ingest_validate import ingest_and_validate_churn

@dsl.pipeline(name="churn-step4-validate-only")
def validate_only_pipeline():
    task = ingest_and_validate_churn(
        s3_endpoint="http://minio.minio.svc.cluster.local:9000",
        bucket="dataset",
        key="churn/v1/raw/train.csv",
        label_column="Churn",
        min_rows=100,
    )

    kubernetes.use_secret_as_env(
        task=task,
        secret_name="minio-connection",
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
        },
    )

if __name__ == "__main__":
    Compiler().compile(
        pipeline_func=validate_only_pipeline,
        package_path="step4_validate_only.yaml",
    )
    print("Compiled step4_validate_only.yaml")
