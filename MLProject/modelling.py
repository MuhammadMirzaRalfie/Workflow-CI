import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# ==============================================
# 0. PARSE ARGUMENTS
# ==============================================
data_path = sys.argv[1] if len(sys.argv) > 1 else "train_preprocessing.csv"
target_var = sys.argv[2] if len(sys.argv) > 2 else "y"

print(f"Loading data from: {data_path}")
print(f"Target variable: {target_var}")
print(f"Current working directory: {os.getcwd()}")


# Configure MLflow tracking/artifact URIs; on Windows avoid file:// scheme
abs_mlruns = os.path.abspath("./mlruns")
# Use a proper file:// URI on all platforms via pathlib.Path.as_uri()
file_uri = Path(abs_mlruns).as_uri()
mlflow.set_tracking_uri(file_uri)
os.environ['MLFLOW_ARTIFACT_ROOT'] = file_uri
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLflow artifact root: {os.environ.get('MLFLOW_ARTIFACT_ROOT')}")


print("Loading dataset...")
df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape}")

X = df.drop(columns=target_var)
y = df[target_var]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# No scaling: use raw features

# No resampling: use the train/test split directly
print("No resampling and no scaling: using train/test split as-is")
X_train_res, y_train_res = X_train, y_train
print(f"Training set shape: {X_train_res.shape}")


print("Training Random Forest model...")
model_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    class_weight='balanced',
    bootstrap=True,
    random_state=42
)

with mlflow.start_run():
    model_rf.fit(X_train_res, y_train_res)
    pred = model_rf.predict(X_test)
    print("Model training completed")

    acc = accuracy_score(y_test, pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, pred))

    print("\n" + "="*50)
    print("Logging to MLflow...")
    print("="*50)

    # Log parameters
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 7)
    mlflow.log_param("min_samples_split", 5)
    mlflow.log_param("min_samples_leaf", 3)
    mlflow.log_param("max_features", "sqrt")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("resampling", "none")
    print("Parameters logged")

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("training_samples", len(X_train_res))
    mlflow.log_metric("test_samples", len(X_test))
    print("Metrics logged")

    print("\nCreating confusion matrix...")
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    # Save to absolute path to avoid issues
    cm_path = os.path.join(os.getcwd(), "confusion_matrix_rf.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    # Log artifact
    try:
        mlflow.log_artifact(cm_path)
        print("Confusion matrix logged to MLflow")
        # Clean up
        if os.path.exists(cm_path):
            os.remove(cm_path)
            print("Temporary file cleaned up")
    except Exception as e:
        print(f"Warning: Could not log confusion matrix: {e}")

    print("\nLogging model to MLflow...")
    try:
        # Log model WITHOUT registered_model_name to avoid registry issues
        mlflow.sklearn.log_model(
            sk_model=model_rf,
            artifact_path="model"
        )
        print("Model successfully logged to MLflow!")
    except Exception as e:
        print(f"ERROR: Failed to log model: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Final Accuracy: {acc:.4f}")
    print(f"Model saved to: {mlflow.get_artifact_uri()}")
    print("="*50)

