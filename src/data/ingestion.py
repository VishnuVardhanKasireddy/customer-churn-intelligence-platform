import pandas as pd
from config.settings import (RAW_DATA_FILE,EXPECTED_COLUMNS,EXPECTED_SHEETS,COLUMN_MAPPING)

def load_retail_data()-> pd.DataFrame:

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset Not Fount : {RAW_DATA_FILE}")

    workbook = pd.ExcelFile(RAW_DATA_FILE)

    missing_sheets = set(EXPECTED_SHEETS) - set(workbook.sheet_names)

    if missing_sheets :
        raise ValueError(f"Missing expected sheets : {sorted(missing_sheets)}")

    frames = []

    for sheet in EXPECTED_SHEETS:
        df = pd.read_excel(workbook,sheet_name=sheet)
        missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing_columns :
            raise ValueError(f"Missing expected columns in {sheet} : {sorted(missing_columns)}")

        df=df.rename(columns=COLUMN_MAPPING)
        frames.append(df)

    combined_df = pd.concat(frames,ignore_index=True)
    combined_df = normalize_dtypes(combined_df)

    return combined_df

def normalize_dtypes(df):

    df["invoice"] = df["invoice"].astype("string")
    df["stock_code"] = df["stock_code"].astype("string")
    df["description"] = df["description"].astype("string")
    df["quantity"] = df["quantity"].astype("int64")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"],errors="coerce")
    df["unit_price"] = df["unit_price"].astype("float64")
    df["country"] = df["country"].astype("string")

    return df