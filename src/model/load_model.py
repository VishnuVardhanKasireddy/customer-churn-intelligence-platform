import joblib

from config.settings import MODEL_FILE


def load_model():
    artifact = joblib.load(MODEL_FILE)

    return artifact


if __name__ == "__main__":
    artifact = load_model()

    print("Model artifact loaded successfully.")
    print(f"Model type: {type(artifact['model']).__name__}")
    print(f"Number of features: {len(artifact['feature_columns'])}")
    print(f"Threshold: {artifact['threshold']}")
    print(f"Model file: {MODEL_FILE}")