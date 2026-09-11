import pandas as pd
from src.features.snapshots import build_churn_snapshots
from src.features.engineering import engineer_customer_features
from config.settings import CLEAN_DATA_FILE,FEATURE_DATA_FILE

def build_features():

    df=pd.read_csv(CLEAN_DATA_FILE,parse_dates=["invoice_date"])

    snapshots = build_churn_snapshots(df)

    features = engineer_customer_features(df,snapshots)

    FEATURE_DATA_FILE.parent.mkdir(parents=True,exist_ok=True)

    features.to_csv(FEATURE_DATA_FILE,index=False)

    return features



if __name__ == "__main__":
    features = build_features()

    print("Feature engineering completed successfully.")
    print(f"Shape: {features.shape}")
    print(f"Output: {FEATURE_DATA_FILE}")