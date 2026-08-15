import pandas as pd
import pytest
from src.data.validation import validate_dataset

@pytest.fixture
def valid_dataframe():
    return pd.DataFrame({
        "invoice":pd.Series(["10001","10002","10003"],dtype="string"),
        "stock_code":pd.Series(["A","B","C"],dtype="string"),
        "description":pd.Series(["Product A","Product B","Product C"],dtype="string"),
        "quantity":pd.Series([1,2,3],dtype="int64"),
        "invoice_date":pd.to_datetime(["2010-01-01","2010-01-02","2010-01-03"]),
        "unit_price":pd.Series([1.0,2.5,4.5],dtype="float64"),
        "customer_id":pd.Series([1001.0,1002.0,1003.0],dtype="float64"),
        "country":pd.Series(["United Kingdom","France","Germany"],dtype="string")
    })

def test_valid_dataframe(valid_dataframe):
    report = validate_dataset(valid_dataframe)

    assert report["valid"] is True
    assert report["errors"] == []

def test_missing_column(valid_dataframe):
    df=valid_dataframe.copy()

    df=df.drop(columns=["customer_id"])

    report = validate_dataset(df)
    assert report["valid"] is False
    assert any("customer_id" in error for error in report["errors"])

def test_extra_column(valid_dataframe):
    df=valid_dataframe.copy()

    df["extra_column"]="unexpected"
    report = validate_dataset(df)

    assert report["valid"] is False
    assert any("extra_column" in error for error in report["errors"])

def test_invalid_dtype(valid_dataframe):
    df=valid_dataframe.copy()
    df["quantity"]=df["quantity"].astype("string")

    report= validate_dataset(df)

    assert report["valid"] is False
    assert any("quantity" in error for error in report["errors"])

def test_missing_values(valid_dataframe):
    df=valid_dataframe.copy()

    df.loc[0,"customer_id"]=None

    report = validate_dataset(df)
    assert report["valid"] is True
    assert report["errors"] == []
    assert report["missing_count"]["customer_id"] == 1
    assert report["missing_percentage"]["customer_id"] > 0

def test_negative_quantity(valid_dataframe):
    df=valid_dataframe.copy()
    df.loc[0,"quantity"]=-5

    report = validate_dataset(df)
    assert report["valid"] is True
    assert report["errors"] == []
    assert report["negative_values"]["quantity"] == 1

def test_negative_price(valid_dataframe):
    df=valid_dataframe.copy()
    df.loc[0,"unit_price"]=-3.5

    report = validate_dataset(df)

    assert report["valid"] is True
    assert report["errors"] == []
    assert report["negative_values"]["unit_price"] ==1

def test_empty_dataset():
    df=pd.DataFrame()

    report = validate_dataset(df)

    assert report["valid"] is False
    assert "Dataset is empty." in report["errors"]

