from kfp import dsl

BASE_IMAGE = "image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/s2i-generic-data-science-notebook:2025.2"

@dsl.component(base_image=BASE_IMAGE)
def ingest_and_validate_churn(
    s3_endpoint: str,
    bucket: str = "dataset",
    key: str = "churn/v1/raw/train.csv",
    label_column: str = "Churn",
    min_rows: int = 100,
    schema_out: dsl.OutputPath(str) = "schema.json",
    profile_out: dsl.OutputPath(str) = "data_profile.json",
):
    import os
    import json
    import boto3
    import pandas as pd

    # --- Contract (Step 3) ---
    required_columns = [
        "customerID","gender","SeniorCitizen","Partner","Dependents","tenure",
        "PhoneService","MultipleLines","InternetService","OnlineSecurity",
        "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
        "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
        "MonthlyCharges","TotalCharges","Churn"
    ]
    allowed_label_values = {"Yes", "No"}

    # --- S3 connection ---
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )

    local_path = "/tmp/train.csv"
    s3.download_file(bucket, key, local_path)

    df = pd.read_csv(local_path)

    # --- Critical validations ---
    if df.shape[0] < min_rows:
        raise ValueError("Dataset too small")

    if len(df.columns) != len(set(df.columns)):
        raise ValueError("Duplicate columns detected")

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    values = set(df[label_column].dropna().unique())
    if not values.issubset(allowed_label_values):
        raise ValueError(f"Invalid label values: {values}")

    # --- Reports ---
    schema = {
        "rows": int(df.shape[0]),
        "label_column": label_column,
        "columns": {c: str(df[c].dtype) for c in df.columns},
    }

    profile = {
        "missing_pct": (df.isna().mean() * 100).to_dict(),
        "label_distribution": df[label_column].value_counts().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    with open(schema_out, "w") as f:
        json.dump(schema, f, indent=2)

    with open(profile_out, "w") as f:
        json.dump(profile, f, indent=2)

    print("Validation passed")
