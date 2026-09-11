import pandas as pd

from src.features.snapshots import (get_valid_cutoff_dates,build_churn_snapshots)

def test_get_valid_cutoff_dates():
    df = pd.DataFrame(
        {
            "invoice_date": pd.date_range(
                "2020-01-01",
                "2021-01-01",
                freq="D",
            )
        }
    )
    cutoffs = get_valid_cutoff_dates(
        df,
        prediction_days=90
    )

    assert len(cutoffs)>0
    assert cutoffs.is_monotonic_increasing

    max_allowed = df["invoice_date"].max()-pd.Timedelta(days=90)

    assert cutoffs.max()<=max_allowed

def test_build_churn_snapshots():
    dates = pd.date_range(
        "2020-01-01",
        "2021-01-01",
        freq="30D",
    )

    df = pd.DataFrame(
        {
            "customer_id": ["C1"] * len(dates),
            "invoice": [f"INV{i}" for i in range(len(dates))],
            "invoice_date": dates,
            "quantity": [1] * len(dates),
            "unit_price": [10.0] * len(dates),
        }
    )

    snapshots = build_churn_snapshots(df,observation_days=180,prediction_days=90)

    assert not snapshots.empty
    assert {
        "customer_id",
        "cutoff_date",
        "churn",
    }.issubset(snapshots.columns)

    assert snapshots["customer_id"].notna().all()
    assert snapshots["churn"].isin([0, 1]).all()