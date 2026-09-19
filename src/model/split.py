import pandas as pd

def temporal_split(df:pd.DataFrame,train_ratio:float = 0.7,validation_ratio:float = 0.15)-> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:

    if df.empty :
        raise ValueError(f"Dataset must not be empty")
    if not 0 < train_ratio < 1:
        raise ValueError(f"train ration must be between 0 and 1")
    if not 0 < validation_ratio < 1 :
        raise ValueError(f"validation ratio must be between 0 and 1")
    if (train_ratio + validation_ratio) >= 1 :
        raise ValueError(f"total spliting ratio must be less than 1")
    if "cutoff_date" not in df.columns :
        raise ValueError(f"cutoff_dates column is not present in dataset")

    data = df.copy()
    data["cutoff_date"] = pd.to_datetime(data["cutoff_date"])
    cutoff_dates = sorted(data["cutoff_date"].unique())

    if len(cutoff_dates) < 3 :
        raise ValueError(f"atleast 3 unique cutoff dates should be present")

    n_dates = len(cutoff_dates)

    train_end = int(n_dates*train_ratio)
    validation_end = int(n_dates*(train_ratio+validation_ratio))

    train_end = max(train_end,1)
    validation_end = max(train_end+1,validation_end)
    validation_end = min(validation_end,n_dates-1)

    train_dates = cutoff_dates[:train_end]
    validation_dates = cutoff_dates[train_end:validation_end]
    test_dates = cutoff_dates[validation_end:]

    train = data[data["cutoff_date"].isin(train_dates)].copy()
    validation = data[data["cutoff_date"].isin(validation_dates)].copy()
    test = data[data["cutoff_date"].isin(test_dates)].copy()

    return train , validation , test 

