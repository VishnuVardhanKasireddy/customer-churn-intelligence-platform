import requests

from dashboard_config import API_BASE_URL


def predict_churn(features: dict) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/api/v1/predict",
        json=features,
        timeout=10,
    )

    response.raise_for_status()

    return response.json()