import pandas as pd

from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "churn"
NON_FEATURE_COLUMNS = {
    "customer_id",
    "cutoff_date",
    TARGET_COLUMN
}

def prepare_data(train : pd.DataFrame, validation : pd.DataFrame, test : pd.DataFrame) :

    for name,df in {
        "train":train,
        "validation":validation,
        "test":test
    }.items():
        if df.empty :
            raise ValueError(f"{name} dataset must not be empty")

        missing_target = TARGET_COLUMN not in df.columns
        if missing_target :
            raise ValueError(f"{name} dataset does not contain {TARGET_COLUMN} target")

    feature_columns = [column for column in train.columns if column not in NON_FEATURE_COLUMNS]

    if not feature_columns : 
        raise ValueError("No feature columns available")

    x_train = train[feature_columns].copy()
    x_validation = validation[feature_columns].copy()
    x_test = test[feature_columns].copy()

    y_train = train[TARGET_COLUMN].copy()
    y_validation = validation[TARGET_COLUMN].copy()
    y_test = test[TARGET_COLUMN].copy()

    return (x_train,x_validation,x_test,y_train,y_validation,y_test,feature_columns)




