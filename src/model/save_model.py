import joblib

from config.settings import MODEL_FILE
from src.model.config import CLASSIFICATION_THRESHOLD
from src.model.train_rf import train_random_forest


def save_model():
    (
        model,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
    ) = train_random_forest()

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "threshold": CLASSIFICATION_THRESHOLD,
    }

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        artifact,
        MODEL_FILE,
    )

    return MODEL_FILE


if __name__ == "__main__":
    model_path = save_model()

    print("Model saved successfully.")
    print(f"Model: {model_path}")