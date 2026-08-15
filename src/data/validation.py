import pandas as pd

def validate_schema(df):
    expected_columns = {
            "invoice",
            "stock_code",
            "description",
            "quantity",
            "invoice_date",
            "unit_price",
            "customer_id",
            "country",
        }

    errors = []
    missing_columns = set(expected_columns) - set(df.columns)
    extra_columns = set(df.columns) - set(expected_columns)

    if(missing_columns):
        errors.append(f"Missing Columns : {sorted(missing_columns)}")
    if(extra_columns):
        errors.append(f"Extra Columns : {sorted(extra_columns)}")

    return errors


def validate_dtypes(df):
    errors = []
    expected_dtypes = {
        "invoice": "string",
        "stock_code": "string",
        "description": "string",
        "quantity": "numeric",
        "invoice_date": "datetime",
        "unit_price": "numeric",
        "customer_id": "numeric",
        "country": "string",
    }

    for column,expected_dtype in expected_dtypes.items():
        if column not in df.columns:
            continue
        series = df[column]
        if expected_dtype == "string":
            valid=pd.api.types.is_string_dtype(series)
        elif expected_dtype == "numeric":
            valid = pd.api.types.is_numeric_dtype(series)
        elif expected_dtype == "datetime":
            valid = pd.api.types.is_datetime64_any_dtype(series)
        else:
            valid = False
        if not valid : 
            errors.append(f"{column} has invalid dtype : {series.dtype}, Expected : {expected_dtype}")

    return errors

def validate_missing_values(df):
    warns = []
    missing_count = df.isna().sum()
    missing_percentage = df.isna().mean()*100

    if missing_count.any():
        columns_with_missing = missing_count[missing_count>0]
        warns.append(f"Missing values detected in columns : {columns_with_missing.to_dict()}")

    return {
        "warnings":warns,
        "missing_count":missing_count.to_dict(),
        "missing_percentage":missing_percentage.to_dict()
    }

def validate_value_range(df):
    warns = []
    negative_values={
        "quantity":0,
        "unit_price":0
    }
    if "quantity" in df.columns and  pd.api.types.is_numeric_dtype(df["quantity"]):
        negative_quantitys = (df["quantity"]<0).sum()
        negative_values["quantity"]=int(negative_quantitys)

        if negative_quantitys > 0:
            warns.append(f"Negative quantities detected : {negative_quantitys}")
    if "unit_price" in df.columns and  pd.api.types.is_numeric_dtype(df["unit_price"]) :
        negative_unit_price = (df["unit_price"]<0).sum()
        negative_values["unit_price"]=int(negative_unit_price)

        if negative_unit_price>0:
            warns.append(f"Negative unit prices detected : {negative_unit_price}")

    return {
        "warnings":warns,
        "negative_values":negative_values
    }


def validate_dataset(df):
    errors = []
    warnings = []

    if df.empty:
        errors.append(f"Dataset is empty.")
        return {
            "valid":False,
            "errors":errors,
            "warnings":warnings,
            "missing_count":0,
            "missing_percentage":0,
            "negative_values":{
                "quantity":0,
                "unit_price":0
            }
        }

    errors.extend(validate_schema(df))
    errors.extend(validate_dtypes(df))
    missing_result=validate_missing_values(df)
    warnings.extend(missing_result["warnings"])
    range_result=validate_value_range(df)
    warnings.extend(range_result["warnings"])

    return {
        "valid": len(errors) == 0,
        "errors":errors,
        "warnings":warnings,
        "missing_count":missing_result["missing_count"],
        "missing_percentage":missing_result["missing_percentage"],
        "negative_values":range_result["negative_values"]
    }


   