import pandas as pd
import pytest

from src.model.preprocessing import (
    prepare_data,
    create_scaler_pipeline,
)


def create_test_data():
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "cutoff_date": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-02-01",
                    "2021-03-01",
                    "2021-04-01",
                ]
            ),
            "total_orders": [1, 3, 5, 2],
            "total_spend": [100.0, 300.0, 800.0, 200.0],
            "recency_days": [100, 50, 10, 80],
            "churn": [1, 0, 0, 1],
        }
    )


def test_prepare_data_separates_target_and_identifiers():
    df = create_test_data()

    result = prepare_data(df, df, df)

    X_train, X_validation, X_test, y_train, y_validation, y_test, feature_columns = result

    assert "churn" not in feature_columns
    assert "customer_id" not in feature_columns
    assert "cutoff_date" not in feature_columns

    assert len(X_train) == len(y_train)
    assert len(X_validation) == len(y_validation)
    assert len(X_test) == len(y_test)


def test_prepare_data_preserves_feature_columns():
    df = create_test_data()

    result = prepare_data(df, df, df)

    feature_columns = result[-1]

    assert feature_columns == [
        "total_orders",
        "total_spend",
        "recency_days",
    ]


def test_prepare_data_rejects_empty_dataset():
    df = create_test_data()
    empty = df.iloc[0:0]

    with pytest.raises(ValueError):
        prepare_data(empty, df, df)


def test_prepare_data_requires_target():
    df = create_test_data().drop(columns=["churn"])

    with pytest.raises(ValueError):
        prepare_data(df, df, df)


def test_scaler_pipeline_standardizes_features():
    df = create_test_data()

    X = df[
        [
            "total_orders",
            "total_spend",
            "recency_days",
        ]
    ]

    pipeline = create_scaler_pipeline()

    transformed = pipeline.fit_transform(X)

    assert transformed.shape == X.shape
    assert transformed.mean(axis=0).round(10).tolist() == [0.0, 0.0, 0.0]