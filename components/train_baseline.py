from kfp import dsl
from kfp.dsl import Output, Artifact, Model, Metrics

BASE_IMAGE = "image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/s2i-generic-data-science-notebook:2025.2"

@dsl.component(base_image=BASE_IMAGE)
def train_baseline_churn(
    s3_endpoint: str,
    bucket: str = "dataset",
    key: str = "churn/v1/raw/train.csv",
    label_column: str = "Churn",
    model: Output[Model] = None,
    preprocessor: Output[Artifact] = None,
    metrics: Output[Metrics] = None,
):
    import os
    import boto3
    import pandas as pd
    import joblib

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    # S3 client
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

    # Prepare features & label
    y = df[label_column].map({"Yes": 1, "No": 0})
    X = df.drop(columns=[label_column, "customerID"])

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    clf = LogisticRegression(max_iter=500)
    pipe = Pipeline(steps=[("preprocessor", preproc), ("classifier", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    # Save artifacts (files)
    joblib.dump(pipe, model.path)
    joblib.dump(preproc, preprocessor.path)

    # Log metrics (metadata-safe)
    metrics.log_metric("accuracy", float(acc))
    metrics.log_metric("roc_auc", float(auc))

    print("Training completed")
    print({"accuracy": acc, "roc_auc": auc})
