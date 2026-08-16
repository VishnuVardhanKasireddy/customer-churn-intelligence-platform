import pandas as pd
import pytest
from src.data.cleaning import classify_transactions
from src.data.cleaning import filter_customer_transactions
from src.data.cleaning import remove_exact_duplicates

@pytest.fixture
def transaction_dataframe():
    return pd.DataFrame({
        "invoice": pd.Series(
            ["10001", "C10002", "10003", "A10004", "10005"],
            dtype="string"
        ),
        "stock_code": pd.Series(
            ["A", "B", "C", "B", "D"],
            dtype="string"
        ),
        "description": pd.Series(
            ["Purchase", "Cancellation", "Return",
             "Bad debt", "Purchase"],
            dtype="string"
        ),
        "quantity": pd.Series(
            [5, -2, -3, 1, 2],
            dtype="int64"
        ),
        "invoice_date": pd.to_datetime([
            "2010-01-01",
            "2010-01-02",
            "2010-01-03",
            "2010-01-04",
            "2010-01-05"
        ]),
        "unit_price": pd.Series(
            [10.0, 10.0, 5.0, -100.0, 20.0],
            dtype="float64"
        ),
        "customer_id": pd.Series(
            [1001.0, 1002.0, 1003.0, None, 1005.0],
            dtype="float64"
        ),
        "country": pd.Series(
            ["UK", "UK", "UK", "UK", "UK"],
            dtype="string"
        ),
    })

def test_purchase_classification(transaction_dataframe):
    result = classify_transactions(transaction_dataframe)

    assert bool(result.loc[0,"is_purchase"] )
    assert not bool(result.loc[0,"is_cancellation"] )
    assert not bool(result.loc[0,"is_operational_negative"] )

def test_cancellation_classification(transaction_dataframe):
    result = classify_transactions(transaction_dataframe)

    assert bool(result.loc[1,"is_cancellation"] )
    assert not bool(result.loc[1,"is_purchase"] )

def test_operational_negative_classification(transaction_dataframe):
    result= classify_transactions(transaction_dataframe)

    assert bool(result.loc[2,"is_operational_negative"] )
    assert not bool(result.loc[2,"is_cancellation"] )
    assert not bool(result.loc[2,"is_purchase"] )

def test_adjustment_classification(transaction_dataframe):
    result = classify_transactions(transaction_dataframe)

    assert bool(result.loc[3,"is_adjustment"] )
    assert not bool(result.loc[3,"is_purchase"])

def test_transaction_value(transaction_dataframe):
    result = classify_transactions(transaction_dataframe)

    assert result.loc[0,"transaction_value"] == 50.0

def test_zero_price_customer_activity(transaction_dataframe):
    df = transaction_dataframe.copy()

    df.loc[0,"unit_price"]=0.0
    df.loc[0,"customer_id"]=1001.0
    result = classify_transactions(df)

    assert bool(result.loc[0,"is_zero_price_activity"])
    assert not bool(result.loc[0,"is_purchase"])

def test_zero_price_anonymous_activity(transaction_dataframe):
    df=transaction_dataframe.copy()

    df.loc[0,"unit_price"] = 0.0
    df.loc[0,"customer_id"] = None
    result=classify_transactions(df)

    assert bool(result.loc[0,"is_zero_price_anonymous"])
    assert  not bool(result.loc[0,"is_purchase"])

def test_zero_quantity(transaction_dataframe):
    df = transaction_dataframe.copy()

    df.loc[0, "quantity"] = 0

    result = classify_transactions(df)

    assert bool(result.loc[0, "is_zero_quantity"])
    assert not bool(result.loc[0, "is_purchase"])

def test_filter_removes_cancellations(transaction_dataframe):
    classified = classify_transactions(transaction_dataframe)
    result = filter_customer_transactions(classified)

    assert not result["is_cancellation"].any()

def test_filter_removes_adjustments(transaction_dataframe):
    classified = classify_transactions(transaction_dataframe)
    result = filter_customer_transactions(classified)

    assert not result["is_adjustment"].any()

def test_filter_removes_operational_negatives(transaction_dataframe):
    classified = classify_transactions(transaction_dataframe)
    result = filter_customer_transactions(classified)

    assert not result["is_operational_negative"].any()

def test_filter_removes_missing_customer_ids(transaction_dataframe):
    classified = classify_transactions(transaction_dataframe)
    result = filter_customer_transactions(classified)

    assert result["customer_id"].notna().all()

def test_filter_keeps_purchases(transaction_dataframe):
    classified = classify_transactions(transaction_dataframe)
    result = filter_customer_transactions(classified)

    assert result["is_purchase"].any()

def test_filter_keeps_zero_price_customer_activity(transaction_dataframe):
    df = transaction_dataframe.copy()

    df.loc[0, "unit_price"] = 0.0
    df.loc[0, "customer_id"] = 1001.0

    classified = classify_transactions(df)
    result = filter_customer_transactions(classified)

    assert result["is_zero_price_activity"].any()

def test_remove_exact_duplicates(transaction_dataframe):
    df = pd.concat(
        [transaction_dataframe, transaction_dataframe],
        ignore_index=True
    )

    result = remove_exact_duplicates(df)

    assert len(result) == len(transaction_dataframe)
    assert not result.duplicated().any()

def test_remove_duplicates_keeps_different_products(transaction_dataframe):
    df = transaction_dataframe.copy()

    df.loc[0, "invoice"] = "10001"
    df.loc[1, "invoice"] = "10001"

    df.loc[0, "stock_code"] = "A"
    df.loc[1, "stock_code"] = "B"

    result = remove_exact_duplicates(df)

    invoice_rows = result[result["invoice"] == "10001"]

    assert len(invoice_rows) == 2

    