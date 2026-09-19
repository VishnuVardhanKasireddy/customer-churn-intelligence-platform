import pandas as pd

from src.model.load_model import load_model


def predict_churn(features: dict) -> dict:
    artifact = load_model()

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    threshold = artifact["threshold"]

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in features
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    input_data = pd.DataFrame(
        [[features[feature] for feature in feature_columns]],
        columns=feature_columns,
    )

    probability = model.predict_proba(input_data)[0, 1]

    prediction = int(probability >= threshold)

    if probability >= 0.70:
        risk_level = "high"
    elif probability >= threshold:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "churn_probability": round(float(probability), 4),
        "churn_prediction": prediction,
        "risk_level": risk_level,
    }