from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


VALID_PAYLOAD = {
    "total_orders": 5,
    "total_spend": 1200,
    "total_quantity": 80,
    "unique_products": 15,
    "active_days": 10,
    "average_order_value": 240,
    "customer_lifespan_days": 120,
    "recency_days": 20,
    "average_interpurchase_days": 25,
    "median_interpurchase_days": 22,
    "orders_30d": 2,
    "orders_60d": 3,
    "orders_90d": 4,
    "spend_30d": 400,
    "spend_60d": 650,
    "spend_90d": 900,
    "orders_previous_90d": 3,
    "spend_previous_90d": 700,
    "order_trend_ratio": 1.33,
    "spend_trend_ratio": 1.29,
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_endpoint():
    response = client.post(
        "/api/v1/predict",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200

    data = response.json()

    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_level" in data

    assert 0 <= data["churn_probability"] <= 1
    assert data["churn_prediction"] in [0, 1]
    assert data["risk_level"] in ["low", "medium", "high"]


def test_prediction_rejects_missing_feature():
    payload = VALID_PAYLOAD.copy()
    del payload["total_orders"]

    response = client.post(
        "/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_prediction_rejects_negative_feature():
    payload = VALID_PAYLOAD.copy()
    payload["total_orders"] = -1

    response = client.post(
        "/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 422