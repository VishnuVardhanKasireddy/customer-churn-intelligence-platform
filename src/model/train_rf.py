import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.model.data import load_model_data
from src.model.config import MODEL_CONFIG

def train_random_forest():
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        feature_columns,
    ) = load_model_data()

    model = RandomForestClassifier(
        **MODEL_CONFIG
    )

    model.fit(X_train, y_train)

    return (
        model,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
    )


if __name__ == "__main__":
    (
        model,
        X_validation,
        y_validation,
        X_test,
        y_test,
        features,
    ) = train_random_forest()

    print("Random Forest training completed successfully.")
    print(f"Number of features: {len(features)}")
    print(f"Validation predictions: {len(X_validation)}")
    print(f"Test predictions: {len(X_test)}")