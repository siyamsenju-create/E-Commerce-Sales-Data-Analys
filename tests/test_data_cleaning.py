"""
Unit tests for data cleaning and validation functions in src/data_cleaning.py.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data_cleaning import (
    add_date_features,
    calculate_unit_price,
    clean_dataset,
    convert_datetime_columns,
    detect_outliers_iqr,
    load_raw_data,
    standardize_column_names,
    validate_data_integrity,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Fixture providing a valid sample DataFrame matching Superstore schema."""
    return pd.DataFrame(
        {
            "Order ID": ["CA-2016-152156", "CA-2016-152156", "CA-2017-138688"],
            "Order Date": ["11/8/2016", "11/8/2016", "6/12/2017"],
            "Ship Date": ["11/11/2016", "11/11/2016", "6/16/2017"],
            "Ship Mode": ["Second Class", "Second Class", "Second Class"],
            "Customer ID": ["CG-12520", "CG-12520", "DV-13045"],
            "Customer Name": ["Claire Gute", "Claire Gute", "Darrin Van Huff"],
            "Segment": ["Consumer", "Consumer", "Corporate"],
            "Country": ["United States", "United States", "United States"],
            "City": ["Henderson", "Henderson", "Los Angeles"],
            "State": ["Kentucky", "Kentucky", "California"],
            "Postal Code": [42420, 42420, 90036],
            "Region": ["South", "South", "West"],
            "Product ID": ["FUR-BO-10001798", "FUR-CH-10000454", "OFF-LA-10000240"],
            "Category": ["Furniture", "Furniture", "Office Supplies"],
            "Sub-Category": ["Bookcases", "Chairs", "Labels"],
            "Product Name": ["Bush Somerset Bookcase", "Hon Deluxe Chair", "Self-Adhesive Labels"],
            "Sales": [261.96, 731.94, 14.62],
            "Quantity": [2, 3, 2],
            "Discount": [0.0, 0.0, 0.0],
            "Profit": [41.91, 219.58, 6.87],
        }
    )


def test_standardize_column_names(sample_raw_df: pd.DataFrame):
    """Test column name standardization to snake_case."""
    cleaned = standardize_column_names(sample_raw_df)
    expected_cols = [
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
    assert list(cleaned.columns) == expected_cols


def test_convert_datetime_columns(sample_raw_df: pd.DataFrame):
    """Test date conversion to datetime64[ns]."""
    cleaned = standardize_column_names(sample_raw_df)
    converted = convert_datetime_columns(cleaned)
    assert pd.api.types.is_datetime64_any_dtype(converted["order_date"])
    assert pd.api.types.is_datetime64_any_dtype(converted["ship_date"])
    assert converted["order_date"].isnull().sum() == 0


def test_add_date_features(sample_raw_df: pd.DataFrame):
    """Test feature engineering of date components and fulfillment duration."""
    cleaned = standardize_column_names(sample_raw_df)
    converted = convert_datetime_columns(cleaned)
    featured = add_date_features(converted)

    assert "order_year" in featured.columns
    assert "order_month" in featured.columns
    assert "order_month_name" in featured.columns
    assert "order_quarter" in featured.columns
    assert "order_year_month" in featured.columns
    assert "shipping_days" in featured.columns

    assert featured.iloc[0]["order_year"] == 2016
    assert featured.iloc[0]["order_month"] == 11
    assert featured.iloc[0]["order_quarter"] == 4
    assert featured.iloc[0]["shipping_days"] == 3


def test_calculate_unit_price(sample_raw_df: pd.DataFrame):
    """Test unit price calculation: unit_price = sales / quantity."""
    cleaned = standardize_column_names(sample_raw_df)
    with_unit = calculate_unit_price(cleaned)
    assert "unit_price" in with_unit.columns
    assert with_unit.iloc[0]["unit_price"] == round(261.96 / 2, 2)
    assert with_unit.iloc[1]["unit_price"] == round(731.94 / 3, 2)


def test_validate_data_integrity_valid(sample_raw_df: pd.DataFrame):
    """Test integrity validation on valid dataframe."""
    cleaned = standardize_column_names(sample_raw_df)
    converted = convert_datetime_columns(cleaned)
    report = validate_data_integrity(converted, raise_on_error=True)

    assert report["row_count"] == 3
    assert report["total_missing_values"] == 0
    assert report["duplicate_rows"] == 0
    assert report["invalid_quantities_count"] == 0
    assert report["negative_sales_count"] == 0
    assert report["invalid_discounts_count"] == 0
    assert report["invalid_shipping_dates_count"] == 0
    assert report["unique_orders"] == 2
    assert report["unique_customers"] == 2


def test_validate_data_integrity_catches_invalid_dates():
    """Test integrity validation catches shipping before order date."""
    df_invalid = pd.DataFrame(
        {
            "order_id": ["O-1"],
            "order_date": pd.to_datetime(["2016-11-10"]),
            "ship_date": pd.to_datetime(["2016-11-05"]),  # Ship date prior to order
            "quantity": [1],
            "sales": [10.0],
            "discount": [0.0],
            "profit": [2.0],
        }
    )
    with pytest.raises(ValueError, match="Data integrity validation failed"):
        validate_data_integrity(df_invalid, raise_on_error=True)


def test_detect_outliers_iqr(sample_raw_df: pd.DataFrame):
    """Test IQR outlier detection returns non-destructive metrics."""
    cleaned = standardize_column_names(sample_raw_df)
    outliers = detect_outliers_iqr(cleaned, numeric_columns=["sales", "profit"])
    assert "sales" in outliers
    assert "profit" in outliers
    assert "iqr" in outliers["sales"]
    assert "lower_bound" in outliers["sales"]
    assert "upper_bound" in outliers["sales"]
    assert "outlier_count" in outliers["sales"]


def test_clean_dataset_pipeline_on_disk(tmp_path: Path):
    """Test end-to-end cleaning pipeline on live project raw dataset."""
    raw_path = Path("data/raw/sales.csv")
    if not raw_path.exists():
        pytest.skip("Live data file data/raw/sales.csv not available")

    out_path = tmp_path / "cleaned_test.csv"
    df_clean, val_report, outlier_report = clean_dataset(
        raw_path=raw_path,
        output_path=out_path,
        raise_on_error=True,
    )

    assert len(df_clean) == 9994
    assert val_report["raw_rows"] == 9994
    assert val_report["cleaned_rows"] == 9994
    assert val_report["total_missing_values"] == 0
    assert val_report["duplicate_rows"] == 0
    assert val_report["unique_orders"] == 5009
    assert val_report["unique_customers"] == 793
    assert out_path.exists()
