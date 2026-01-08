from kfp import dsl
from kfp.compiler import Compiler
from kfp import kubernetes

from components.train_baseline import train_baseline_churn
from components.save_model_to_minio import save_model_to_minio

@dsl.pipeline(name="churn-step11-train-and-save")
def train_and_save_pipeline():
    train = train_baseline_churn(
        s3_endpoint="http://minio.minio.svc.cluster.local:9000",
        bucket="dataset",
        key="churn/v1/raw/train.csv",
        label_column="Churn",
    )

    # Use pipeline run name as a simple version tag
    run_id = dsl.PIPELINE_JOB_NAME_PLACEHOLDER

    save = save_model_to_minio(
        s3_endpoint="http://minio.minio.svc.cluster.local:9000",
        models_bucket="models",
        model_prefix=f"churn/baseline/{run_id}",
        trained_model=train.outputs["model"],
        preprocessor=train.outputs["preprocessor"],
        metrics=train.outputs["metrics"],
    )

    # Inject MinIO creds into both steps
    for t in [train, save]:
        kubernetes.use_secret_as_env(
            task=t,
            secret_name="minio-connection",
            secret_key_to_env={
                "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            },
        )

if __name__ == "__main__":
    Compiler().compile(
        pipeline_func=train_and_save_pipeline,
        package_path="step11_train_and_save.yaml",
    )
    print("Compiled step11_train_and_save.yaml")
