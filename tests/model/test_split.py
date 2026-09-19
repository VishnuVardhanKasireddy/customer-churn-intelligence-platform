import pandas as pd
import pytest

from src.model.split import temporal_split

def create_test_data():
    dates = pd.date_range(
        "2021-01-01",
        periods=10,
        freq="MS",
    )

    return pd.DataFrame(
        {
            "customer_id": range(10),
            "cutoff_date": dates,
            "feature": range(10),
            "churn": [0, 1] * 5,
        }
    )


def test_temporal_split_preserves_chronological_order():
    df = create_test_data()

    train, validation, test = temporal_split(df)

    assert train["cutoff_date"].max() < validation["cutoff_date"].min()
    assert validation["cutoff_date"].max() < test["cutoff_date"].min()


def test_temporal_split_contains_all_rows():
    df = create_test_data()

    train, validation, test = temporal_split(df)

    assert len(train) + len(validation) + len(test) == len(df)


def test_temporal_split_contains_all_dates():
    df = create_test_data()

    train, validation, test = temporal_split(df)

    split_dates = set(train["cutoff_date"])
    split_dates.update(validation["cutoff_date"])
    split_dates.update(test["cutoff_date"])

    assert split_dates == set(df["cutoff_date"])


def test_temporal_split_has_no_overlap():
    df = create_test_data()

    train, validation, test = temporal_split(df)

    train_dates = set(train["cutoff_date"])
    validation_dates = set(validation["cutoff_date"])
    test_dates = set(test["cutoff_date"])

    assert train_dates.isdisjoint(validation_dates)
    assert train_dates.isdisjoint(test_dates)
    assert validation_dates.isdisjoint(test_dates)


def test_temporal_split_rejects_invalid_ratios():
    df = create_test_data()

    with pytest.raises(ValueError):
        temporal_split(df, train_ratio=0.8, validation_ratio=0.3)


def test_temporal_split_requires_cutoff_date():
    df = create_test_data().drop(columns=["cutoff_date"])

    with pytest.raises(ValueError):
        temporal_split(df)