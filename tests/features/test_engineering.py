import pandas as pd

from src.features.engineering import engineer_customer_features


def create_test_data():
    return pd.DataFrame(
        {
            "customer_id": [1, 1, 1, 2, 2],
            "invoice": ["I1", "I1", "I2", "I3", "I4"],
            "invoice_date": pd.to_datetime(
                [
                    "2020-01-10",
                    "2020-01-10",
                    "2020-03-10",
                    "2020-02-10",
                    "2020-05-10",
                ]
            ),
            "quantity": [2, 3, 5, 2, 4],
            "unit_price": [10.0, 10.0, 20.0, 15.0, 25.0],
            "stock_code": ["P1", "P2", "P1", "P3", "P4"],
        }
    )


def create_test_snapshots():
    return pd.DataFrame(
        {
            "customer_id": [1, 2],
            "cutoff_date": pd.to_datetime(
                ["2020-03-15", "2020-05-15"]
            ),
            "churn": [0, 1],
        }
    )


def test_customer_features_are_created():
    df = create_test_data()
    snapshots = create_test_snapshots()

    features = engineer_customer_features(
        df,
        snapshots,
        observation_days=180,
    )

    expected_columns = {
        "customer_id",
        "cutoff_date",
        "total_orders",
        "total_spend",
        "total_quantity",
        "unique_products",
        "active_days",
        "average_order_value",
        "customer_lifespan_days",
        "recency_days",
        "average_interpurchase_days",
        "median_interpurchase_days",
        "orders_30d",
        "orders_60d",
        "orders_90d",
        "spend_30d",
        "spend_60d",
        "spend_90d",
        "orders_previous_90d",
        "spend_previous_90d",
        "order_trend_ratio",
        "spend_trend_ratio",
        "churn",
    }

    assert expected_columns.issubset(features.columns)


def test_customer_features_have_valid_values():
    df = create_test_data()
    snapshots = create_test_snapshots()

    features = engineer_customer_features(
        df,
        snapshots,
        observation_days=180,
    )

    assert features["customer_id"].notna().all()
    assert (features["total_orders"] > 0).all()
    assert (features["total_spend"] >= 0).all()
    assert (features["total_quantity"] > 0).all()
    assert (features["unique_products"] > 0).all()
    assert (features["active_days"] > 0).all()
    assert (features["recency_days"] >= 0).all()
    assert features["churn"].isin([0, 1]).all()


def test_average_order_value_is_correct():
    df = create_test_data()
    snapshots = create_test_snapshots()

    features = engineer_customer_features(
        df,
        snapshots,
        observation_days=180,
    )

    customer_1 = features[
        features["customer_id"] == 1
    ].iloc[0]

    assert customer_1["total_orders"] == 2
    assert customer_1["total_spend"] == 150
    assert customer_1["average_order_value"] == 75


def test_customer_lifespan_is_correct():
    df = create_test_data()
    snapshots = create_test_snapshots()

    features = engineer_customer_features(
        df,
        snapshots,
        observation_days=180,
    )

    customer_1 = features[
        features["customer_id"] == 1
    ].iloc[0]

    assert customer_1["customer_lifespan_days"] == 60


def test_churn_labels_are_preserved():
    df = create_test_data()
    snapshots = create_test_snapshots()

    features = engineer_customer_features(
        df,
        snapshots,
        observation_days=180,
    )

    assert features.set_index("customer_id")["churn"].to_dict() == {
        1: 0,
        2: 1,
    }

def test_recent_and_trend_features_are_valid():
    df = create_test_data()
    snapshots = create_test_snapshots()

    features = engineer_customer_features(
        df,
        snapshots,
        observation_days=180,
    )

    assert (features["orders_30d"] >= 0).all()
    assert (features["orders_60d"] >= 0).all()
    assert (features["orders_90d"] >= 0).all()

    assert (features["spend_30d"] >= 0).all()
    assert (features["spend_60d"] >= 0).all()
    assert (features["spend_90d"] >= 0).all()

    assert (features["orders_previous_90d"] >= 0).all()
    assert (features["spend_previous_90d"] >= 0).all()

    assert (features["order_trend_ratio"] >= 0).all()
    assert (features["spend_trend_ratio"] >= 0).all()

def test_features_do_not_use_future_transactions():
    df = create_test_data()

    snapshots = create_test_snapshots()

    features = engineer_customer_features(
        df,
        snapshots,
        observation_days=180,
    )

    for _, row in features.iterrows():
        customer_transactions = df[
            df["customer_id"] == row["customer_id"]
        ]

        future_transactions = customer_transactions[
            customer_transactions["invoice_date"] > row["cutoff_date"]
        ]

        # Feature calculations must never depend on future transactions.
        # Therefore, changing future transactions should not change
        # the historical feature values.
        historical_df = df.copy()

        historical_df.loc[
            historical_df["invoice_date"] > row["cutoff_date"],
            "quantity",
        ] = 999999

        historical_features = engineer_customer_features(
            historical_df,
            snapshots[
                snapshots["customer_id"] == row["customer_id"]
            ],
            observation_days=180,
        )

        original = features[
            (features["customer_id"] == row["customer_id"])
            & (features["cutoff_date"] == row["cutoff_date"])
        ].iloc[0]

        modified = historical_features[
            historical_features["cutoff_date"] == row["cutoff_date"]
        ].iloc[0]

        feature_columns = [
            "total_orders",
            "total_spend",
            "total_quantity",
            "unique_products",
            "active_days",
            "average_order_value",
            "customer_lifespan_days",
            "recency_days",
            "average_interpurchase_days",
            "median_interpurchase_days",
            "orders_30d",
            "orders_60d",
            "orders_90d",
            "spend_30d",
            "spend_60d",
            "spend_90d",
            "orders_previous_90d",
            "spend_previous_90d",
            "order_trend_ratio",
            "spend_trend_ratio",
        ]

        for column in feature_columns:
            assert original[column] == modified[column]