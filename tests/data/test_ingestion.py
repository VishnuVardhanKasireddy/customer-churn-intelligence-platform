import pandas as pd
from src.data.ingestion import load_retail_data

def test_load_retail_data_returns_dataframe():
    df = load_retail_data()
    assert isinstance(df,pd.DataFrame)

def test_load_retail_data_has_expected_columns():
    df = load_retail_data()

    expected_columns = {
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "unit_price",
        "customer_id",
        "country",
    }

    assert set(df.columns) == expected_columns

def test_load_retail_data_contains_both_periods():
    df = load_retail_data()

    assert df["invoice_date"].min().year == 2009
    assert df["invoice_date"].max().year == 2011