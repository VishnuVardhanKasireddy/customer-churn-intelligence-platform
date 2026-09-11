from src.data.ingestion import load_retail_data
from src.data.validation import validate_dataset
from src.data.cleaning import clean_transactions
from config.settings import CLEAN_DATA_FILE

def process_data():

    df = load_retail_data()

    validation_report = validate_dataset(df)

    if not validation_report["valid"] :
        raise ValueError(f"Dataset validation failed : {validation_report['errors']}")

    clean_df = clean_transactions(df)

    CLEAN_DATA_FILE.parent.mkdir(parents=True,exist_ok=True)

    clean_df.to_csv(CLEAN_DATA_FILE,index=False)

    print(f"Processed data saved to: {CLEAN_DATA_FILE}")
    print(f"Rows: {len(clean_df):,}")
    print(f"Customers: {clean_df['customer_id'].nunique():,}")
    
    return clean_df


