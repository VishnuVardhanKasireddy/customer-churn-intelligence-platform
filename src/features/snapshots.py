import pandas as pd

OBSERVATION_DAYS = 180
PREDICTION_DAYS = 90

def get_valid_cutoff_dates(df:pd.DataFrame,prediction_days : int = PREDICTION_DAYS)->pd.DatetimeIndex:

    min_date = df["invoice_date"].min()
    max_date = df["invoice_date"].max()

    latest_cutoff = max_date - pd.Timedelta(days=prediction_days)

    cutoffs = pd.date_range(
        start=min_date + pd.Timedelta(days= OBSERVATION_DAYS),
        end=latest_cutoff,
        freq="MS"
    )

    return cutoffs

def build_churn_snapshots(df:pd.DataFrame,observation_days:int = OBSERVATION_DAYS,prediction_days:int = PREDICTION_DAYS)->pd.DataFrame:

    required_columns={
        "customer_id",
        "invoice",
        "invoice_date",
        "quantity",
        "unit_price"
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns : {sorted(missing_columns)}"
        )

    data = df.copy()
    data["invoice"] = data["invoice"].astype("string")
    data["invoice_date"] = pd.to_datetime(data["invoice_date"])

    purchases = data[
        (data["customer_id"].notna())
        & (data["quantity"]>0)
        & (data["unit_price"]>0)
        & (~data["invoice"].str.upper().str.startswith("C"))
    ].copy()

    cutoffs = get_valid_cutoff_dates(purchases,prediction_days=prediction_days)

    snapshots = []

    for cutoff in cutoffs :

        observation_start = cutoff - pd.Timedelta(days=observation_days)
        prediction_end = cutoff + pd.Timedelta(days=prediction_days)

        historical = purchases[
            (purchases["invoice_date"]>observation_start)
            &(purchases["invoice_date"]<=cutoff)
        ]

        future = purchases[
            (purchases["invoice_date"]>cutoff)
            &(purchases["invoice_date"]<=prediction_end)
        ]

        customers = historical["customer_id"].unique()
        if len(customers)==0:
            continue

        future_customers = set(future["customer_id"].unique())

        snapshot = pd.DataFrame(
            {
                "customer_id":customers,
                "cutoff_date":cutoff
            }
        )
        snapshot["churn"] = (~snapshot["customer_id"].isin(future_customers)).astype(int)

        snapshots.append(snapshot)

    if not snapshots:
        raise ValueError("No valid customer snapshots created")

    return pd.concat(snapshots,ignore_index=True)