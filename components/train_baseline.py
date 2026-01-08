from kfp import dsl

BASE_IMAGE = "image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/s2i-generic-data-science-notebook:2025.2"

@dsl.component(base_image=BASE_IMAGE)
def train_baseline_churn(
    s3_endpoint: str,
    bucket: str = "dataset",
    key: str = "churn/v1/raw/train.csv",
    label_column: str = "Churn",
    model_out: dsl.OutputPath(str) = "model.joblib",
    preprocessor_out: dsl.OutputPath(str) = "preprocessor.joblib",
    metrics_out: dsl.OutputPath(str) = "metrics.json",
):
    import os
    import json
    import boto3
    import pandas as pd
    import joblib

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    # --- Load data from MinIO ---
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

    # --- Prepare features & label ---
    y = df[label_column].map({"Yes": 1, "No": 0})
    X = df.drop(columns=[label_column, "customerID"])

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = LogisticRegression(max_iter=500)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    # --- Train / validation split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)

    # --- Evaluate ---
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }

    # --- Save artifacts ---
    joblib.dump(pipeline, model_out)
    joblib.dump(preprocessor, preprocessor_out)

    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training completed")
    print(metrics)
