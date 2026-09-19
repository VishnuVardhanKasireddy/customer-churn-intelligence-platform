import pytest

from src.model.predict import predict_churn


VALID_FEATURES = {
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


def test_prediction_returns_expected_keys():
    result = predict_churn(VALID_FEATURES)

    assert "churn_probability" in result
    assert "churn_prediction" in result
    assert "risk_level" in result


def test_probability_is_valid():
    result = predict_churn(VALID_FEATURES)

    assert 0 <= result["churn_probability"] <= 1


def test_prediction_is_binary():
    result = predict_churn(VALID_FEATURES)

    assert result["churn_prediction"] in {0, 1}


def test_risk_level_is_valid():
    result = predict_churn(VALID_FEATURES)

    assert result["risk_level"] in {"low", "medium", "high"}


def test_missing_feature_is_rejected():
    features = VALID_FEATURES.copy()
    del features["recency_days"]

    with pytest.raises(ValueError, match="Missing required features"):
        predict_churn(features)