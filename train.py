import pandas as pd
import sys
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json

# Command-line arguments
input_file, model_file, metrics_file = sys.argv[1], sys.argv[2], sys.argv[3]

# Load dataset
df = pd.read_csv(input_file)
X = df.drop(columns=["species"])
y = df["species"]

# Split into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow run
with mlflow.start_run():
    model = LogisticRegression(max_iter=200)

    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 200)

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", report["weighted avg"]["precision"])
    mlflow.log_metric("recall", report["weighted avg"]["recall"])
    mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

    # Save model
    joblib.dump(model, model_file)
    mlflow.sklearn.log_model(model, "model")

    # Save accuracy to JSON for DVC metrics
    metrics = {
        "accuracy": accuracy,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"]
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
