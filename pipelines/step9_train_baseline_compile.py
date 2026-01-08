from kfp import dsl
from kfp.compiler import Compiler
from kfp import kubernetes

from components.train_baseline import train_baseline_churn

@dsl.pipeline(name="churn-step9-train-baseline")
def train_baseline_pipeline():
    task = train_baseline_churn(
        s3_endpoint="http://minio.minio.svc.cluster.local:9000",
        bucket="dataset",
        key="churn/v1/raw/train.csv",
        label_column="Churn",
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
        pipeline_func=train_baseline_pipeline,
        package_path="step9_train_baseline.yaml",
    )
    print("Compiled step9_train_baseline.yaml")
