import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def run(data_path):
    # Validasi file ada
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with mlflow.start_run():
        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

        # Pastikan kolom target benar
        if 'y' not in df.columns:
            raise ValueError("Target column 'y' not found in dataset")

        y = df['y']
        X = df.drop(columns=['y'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="train_preprocessing.csv")
    args = parser.parse_args()

    run(args.data_path)
