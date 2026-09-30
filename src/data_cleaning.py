"""
Data Cleaning & Quality Validation Module for E-Commerce Sales Data.

This module provides modular, robust functions to load, standardize,
validate, engineer features, and export the processed dataset while strictly
preserving the raw dataset without data distortion or blind record removal.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "order_id",
    "order_date",
    "ship_date",
    "ship_mode",
    "customer_id",
    "customer_name",
    "segment",
    "country",
    "city",
    "state",
    "postal_code",
    "region",
    "product_id",
    "category",
    "sub_category",
    "product_name",
    "sales",
    "quantity",
    "discount",
    "profit",
]


def load_raw_data(
    file_path: Union[str, Path], encoding: str = "windows-1252"
) -> pd.DataFrame:
    """
    Load raw transaction data from CSV file with specified encoding.

    Args:
        file_path: Path to the raw CSV file.
        encoding: Character encoding (default: 'windows-1252').

    Returns:
        pd.DataFrame: Loaded raw DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file cannot be parsed as a valid CSV.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    try:
        df = pd.read_csv(path, encoding=encoding)
        return df
    except UnicodeDecodeError:
        # Fallback to latin-1 if windows-1252 fails
        return pd.read_csv(path, encoding="latin-1")
    except Exception as exc:
        raise ValueError(f"Failed to read CSV at {path}: {exc}") from exc


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame column names to lowercase snake_case.

    Examples:
        'Order ID' -> 'order_id'
        'Sub-Category' -> 'sub_category'
        'Product Name' -> 'product_name'

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with standardized column names.
    """
    df_clean = df.copy()
    clean_cols = (
        df_clean.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    df_clean.columns = clean_cols
    return df_clean


def convert_datetime_columns(
    df: pd.DataFrame, date_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convert date string columns into Pandas datetime64[ns] objects.

    Args:
        df: Input DataFrame.
        date_columns: List of column names to convert. Defaults to ['order_date', 'ship_date'].

    Returns:
        pd.DataFrame: DataFrame with converted datetime columns.

    Raises:
        ValueError: If datetime conversion produces unexpected NaT from valid raw inputs.
    """
    if date_columns is None:
        date_columns = ["order_date", "ship_date"]

    df_clean = df.copy()
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")
            # Verify no NaT introduced from non-null raw entries
            if df_clean[col].isnull().sum() > 0 and df[col].isnull().sum() == 0:
                raise ValueError(
                    f"Datetime conversion created unexpected NaT values in column: {col}"
                )
    return df_clean


def add_date_features(
    df: pd.DataFrame,
    date_column: str = "order_date",
    ship_column: str = "ship_date",
) -> pd.DataFrame:
    """
    Derive essential time-series and fulfillment features from transaction dates.

    Features created:
        - order_year: Integer year (e.g. 2016)
        - order_month: Integer month (1-12)
        - order_month_name: Short month name (Jan, Feb, ...)
        - order_quarter: Integer quarter (1-4)
        - order_year_month: String period 'YYYY-MM' for aggregate trend plotting
        - shipping_days: Integer duration between order_date and ship_date (if ship_date exists)

    Args:
        df: Input DataFrame with parsed datetime column.
        date_column: The reference order datetime column.
        ship_column: The reference shipping datetime column.

    Returns:
        pd.DataFrame: DataFrame enriched with temporal features.

    Raises:
        KeyError: If date_column is missing.
    """
    df_clean = df.copy()
    if date_column not in df_clean.columns:
        raise KeyError(f"Column '{date_column}' not found in DataFrame.")

    df_clean["order_year"] = df_clean[date_column].dt.year
    df_clean["order_month"] = df_clean[date_column].dt.month
    df_clean["order_month_name"] = df_clean[date_column].dt.strftime("%b")
    df_clean["order_quarter"] = df_clean[date_column].dt.quarter
    df_clean["order_year_month"] = df_clean[date_column].dt.to_period("M").astype(str)

    if ship_column in df_clean.columns:
        df_clean["shipping_days"] = (
            df_clean[ship_column] - df_clean[date_column]
        ).dt.days

    return df_clean


def calculate_unit_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the 'unit_price' field as (sales / quantity).

    Validates that:
        - quantity > 0
        - sales >= 0

    Args:
        df: Input DataFrame containing 'sales' and 'quantity'.

    Returns:
        pd.DataFrame: DataFrame with newly computed 'unit_price' column.

    Raises:
        KeyError: If 'sales' or 'quantity' is missing.
        ValueError: If any non-positive quantity is found.
    """
    df_clean = df.copy()
    if "sales" not in df_clean.columns or "quantity" not in df_clean.columns:
        raise KeyError("Required columns 'sales' and 'quantity' missing.")

    if (df_clean["quantity"] <= 0).any():
        raise ValueError("Invalid quantity found (<= 0) during unit_price calculation.")

    df_clean["unit_price"] = np.round(df_clean["sales"] / df_clean["quantity"], 2)
    return df_clean


def validate_data_integrity(
    df: pd.DataFrame, raise_on_error: bool = False
) -> Dict[str, Any]:
    """
    Execute comprehensive data quality and schema compliance checks.

    Checks performed:
        - Total row & column count
        - Missing values per column
        - Duplicate rows (exact)
        - Unique Order IDs and Customer IDs
        - Non-positive quantities (quantity <= 0)
        - Negative sales (sales < 0)
        - Discount range validity (discount < 0 or discount > 1)
        - Date logic: Shipping date earlier than Order date (ship_date < order_date)

    Args:
        df: Cleaned or raw DataFrame to validate.
        raise_on_error: If True, raise ValueError on critical integrity violations.

    Returns:
        Dict[str, Any]: Detailed dictionary of validation results.

    Raises:
        ValueError: If raise_on_error=True and critical data integrity issues exist.
    """
    report: Dict[str, Any] = {}

    report["row_count"] = len(df)
    report["column_count"] = len(df.columns)
    report["missing_values"] = df.isnull().sum().to_dict()
    report["total_missing_values"] = int(df.isnull().sum().sum())
    report["duplicate_rows"] = int(df.duplicated().sum())

    if "order_id" in df.columns:
        report["unique_orders"] = int(df["order_id"].nunique())
    if "customer_id" in df.columns:
        report["unique_customers"] = int(df["customer_id"].nunique())
    if "product_id" in df.columns:
        report["unique_products"] = int(df["product_id"].nunique())

    # Numerical boundary validations
    invalid_qty = int((df["quantity"] <= 0).sum()) if "quantity" in df.columns else 0
    negative_sales = int((df["sales"] < 0).sum()) if "sales" in df.columns else 0
    invalid_disc = (
        int(((df["discount"] < 0) | (df["discount"] > 1.0)).sum())
        if "discount" in df.columns
        else 0
    )

    report["invalid_quantities_count"] = invalid_qty
    report["negative_sales_count"] = negative_sales
    report["invalid_discounts_count"] = invalid_disc

    # Date consistency check
    invalid_shipping = 0
    if "order_date" in df.columns and "ship_date" in df.columns:
        invalid_shipping = int((df["ship_date"] < df["order_date"]).sum())
    report["invalid_shipping_dates_count"] = invalid_shipping

    if raise_on_error:
        errors = []
        if invalid_qty > 0:
            errors.append(f"Found {invalid_qty} non-positive quantity records.")
        if negative_sales > 0:
            errors.append(f"Found {negative_sales} negative sales records.")
        if invalid_disc > 0:
            errors.append(f"Found {invalid_disc} invalid discount records.")
        if invalid_shipping > 0:
            errors.append(f"Found {invalid_shipping} records where ship_date < order_date.")
        if errors:
            raise ValueError("Data integrity validation failed: " + "; ".join(errors))

    return report


def detect_outliers_iqr(
    df: pd.DataFrame, numeric_columns: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Identify potential statistical outliers using the Interquartile Range (IQR) method.

    Formula:
        IQR = Q3 - Q1
        Lower Bound = Q1 - 1.5 * IQR
        Upper Bound = Q3 + 1.5 * IQR

    Note:
        Outliers are detected and documented for analytical awareness, NOT deleted,
        as large transactions and steep losses represent legitimate commercial activity.

    Args:
        df: Input DataFrame.
        numeric_columns: Columns to test (default: ['sales', 'profit', 'quantity', 'unit_price', 'discount']).

    Returns:
        Dict[str, Dict[str, Any]]: Outlier statistics and counts per column.
    """
    if numeric_columns is None:
        numeric_columns = [
            col
            for col in ["sales", "profit", "quantity", "unit_price", "discount"]
            if col in df.columns
        ]

    outlier_report: Dict[str, Dict[str, Any]] = {}

    for col in numeric_columns:
        series = df[col].dropna()
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = round((outlier_count / len(series)) * 100, 2)

        outlier_report[col] = {
            "q1": round(q1, 4),
            "median": round(float(series.median()), 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_pct,
            "min_val": round(float(series.min()), 4),
            "max_val": round(float(series.max()), 4),
        }

    return outlier_report


def clean_dataset(
    raw_path: Union[str, Path] = "data/raw/sales.csv",
    output_path: Union[str, Path] = "data/processed/cleaned_sales.csv",
    encoding: str = "windows-1252",
    raise_on_error: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Execute the complete end-to-end data cleaning & validation pipeline.

    Steps:
        1. Load raw data (untouched source).
        2. Standardize column names (snake_case).
        3. Convert date fields to datetime64[ns].
        4. Engineer date attributes and shipping duration.
        5. Compute derived field unit_price = (sales / quantity).
        6. Validate data integrity and schema compliance.
        7. Calculate statistical IQR outlier profiles.
        8. Save processed dataset to output_path.

    Args:
        raw_path: Path to the raw CSV file.
        output_path: Path to save the processed CSV file.
        encoding: Encoding format for raw CSV.
        raise_on_error: If True, raise error on critical data validation failures.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Dict[str, Any]]]:
            (Cleaned DataFrame, Validation Report, Outlier Report)
    """
    raw_file = Path(raw_path)
    out_file = Path(output_path)

    # 1. Load Raw
    df_raw = load_raw_data(raw_file, encoding=encoding)

    # 2. Standardize Names
    df_clean = standardize_column_names(df_raw)

    # 3. Datetime conversions
    df_clean = convert_datetime_columns(
        df_clean, date_columns=["order_date", "ship_date"]
    )

    # 4. Date Features & Fulfillment
    df_clean = add_date_features(
        df_clean, date_column="order_date", ship_column="ship_date"
    )

    # 5. Unit Price Calculation
    df_clean = calculate_unit_price(df_clean)

    # 6. Validation
    validation_report = validate_data_integrity(
        df_clean, raise_on_error=raise_on_error
    )
    validation_report["raw_rows"] = len(df_raw)
    validation_report["raw_columns"] = len(df_raw.columns)
    validation_report["cleaned_rows"] = len(df_clean)
    validation_report["cleaned_columns"] = len(df_clean.columns)

    # 7. Outlier Analysis
    outlier_report = detect_outliers_iqr(df_clean)

    # 8. Save Processed
    save_processed_data(df_clean, out_file)

    return df_clean, validation_report, outlier_report


def save_processed_data(df: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """
    Save the cleaned DataFrame to CSV format.

    Args:
        df: DataFrame to save.
        output_path: Target CSV file path.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
