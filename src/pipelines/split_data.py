import pandas as pd

from config.settings import (FEATURE_DATA_FILE,TRAIN_DATA_FILE,VALIDATION_DATA_FILE,TEST_DATA_FILE)
from src.model.split import temporal_split

def split_data():
    df=pd.read_csv(FEATURE_DATA_FILE,parse_dates=["cutoff_date"])

    train,validation,test = temporal_split(df)

    TRAIN_DATA_FILE.parent.mkdir(parents=True,exist_ok=True)

    train.to_csv(TRAIN_DATA_FILE,index=False)
    validation.to_csv(VALIDATION_DATA_FILE,index=False)
    test.to_csv(TEST_DATA_FILE,index=False)

    return train,validation,test

if __name__ == "__main__":
    train, validation, test = split_data()

    print("Temporal split completed successfully.")
    print(f"Train:      {train.shape}")
    print(f"Validation: {validation.shape}")
    print(f"Test:       {test.shape}")

    print(
        f"Train dates: "
        f"{train['cutoff_date'].min()} → "
        f"{train['cutoff_date'].max()}"
    )

    print(
        f"Validation dates: "
        f"{validation['cutoff_date'].min()} → "
        f"{validation['cutoff_date'].max()}"
    )

    print(
        f"Test dates: "
        f"{test['cutoff_date'].min()} → "
        f"{test['cutoff_date'].max()}"
    )