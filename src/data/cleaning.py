import pandas as pd

def classify_transactions(df) : 
    result = df.copy()

    result["is_cancellation"]=(result["invoice"].str.upper().str.startswith("C"))

    result["is_adjustment"] = ((result["invoice"].str.upper().str.startswith("A")) & (result["stock_code"].eq("B")))

    result["is_operational_negative"] = (
        (result["quantity"] < 0 )&
        (~result["is_cancellation"]) &
        (~result["is_adjustment"])
    )

    result["is_zero_quantity"] = result["quantity"].eq(0)

    result["is_zero_price_activity"] = (
        (result["quantity"] > 0)  &
        (result["unit_price"] == 0) &
        (result["customer_id"].notna()) &
        (~result["is_cancellation"]) &
        (~result["is_adjustment"])
    )

    result["is_zero_price_anonymous"] = (
        (result["quantity"] > 0) &
        (result["unit_price"] == 0) &
        (result["customer_id"].isna()) &
        (~result["is_cancellation"]) &
        (~result["is_adjustment"])
    )

    result["is_purchase"] = (
        (result["quantity"] > 0) &
        (result["unit_price"] > 0) &
        (~result["is_cancellation"]) &
        (~result["is_adjustment"])
    )

    result["transaction_value"] = (result["quantity"] * result["unit_price"])

    return result

def filter_customer_transactions(df):
    keep_mask=(
       ( df["is_purchase"]
        | df["is_zero_price_activity"]) &
        (df["customer_id"].notna())
    )

    return df.loc[keep_mask].copy()

def remove_exact_duplicates(df):
    return df.drop_duplicates().copy()


def clean_transactions(df):
    classified_df = classify_transactions(df)
    filtered_df = filter_customer_transactions(classified_df)
    deduplicated = remove_exact_duplicates(filtered_df)

    return deduplicated