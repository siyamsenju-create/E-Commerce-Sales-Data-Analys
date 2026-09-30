"""
Unit tests for edge cases, boundary handling, and anomaly resilience.
"""

import pandas as pd
import pytest

from src.analysis import (
    calculate_category_metrics,
    calculate_core_kpis,
    calculate_correlations,
    calculate_customer_metrics,
    calculate_distribution_statistics,
    calculate_product_metrics,
    calculate_regional_metrics,
)
from src.data_cleaning import (
    calculate_unit_price,
    detect_outliers_iqr,
    standardize_column_names,
    validate_data_integrity,
)


def test_empty_dataframe_kpis():
    """Verify calculate_core_kpis handles empty DataFrame without crashing."""
    empty_df = pd.DataFrame(
        columns=["sales", "profit", "quantity", "order_id", "customer_id"]
    )
    kpis = calculate_core_kpis(empty_df)
    assert kpis["total_revenue"] == 0.0
    assert kpis["total_orders"] == 0
    assert kpis["avg_order_value"] == 0.0
    assert kpis["profitable_orders_count"] == 0


def test_empty_dataframe_analytics():
    """Verify group-by and correlation functions handle empty DataFrames gracefully."""
    empty_df = pd.DataFrame(
        columns=[
            "sales",
            "profit",
            "category",
            "region",
            "discount",
            "quantity",
            "unit_price",
        ]
    )
    assert calculate_category_metrics(empty_df).empty
    assert calculate_regional_metrics(empty_df).empty
    assert calculate_correlations(empty_df).empty
    assert calculate_distribution_statistics(empty_df).empty

    cust_dist, top_cust = calculate_customer_metrics(empty_df)
    assert cust_dist.empty
    assert top_cust.empty


def test_unit_price_zero_or_negative_quantity_raises_error():
    """Verify calculate_unit_price rejects invalid non-positive quantities."""
    df_bad_qty = pd.DataFrame({"sales": [100.0, 50.0], "quantity": [0, -2]})
    with pytest.raises(ValueError, match="Invalid quantity found"):
        calculate_unit_price(df_bad_qty)


def test_unit_price_missing_columns_raises_error():
    """Verify calculate_unit_price raises KeyError when required columns are missing."""
    df_missing = pd.DataFrame({"revenue": [100.0], "units": [2]})
    with pytest.raises(KeyError):
        calculate_unit_price(df_missing)


def test_validate_data_integrity_detects_all_anomalies():
    """Verify validate_data_integrity detects nulls, duplicates, negative sales, and out-of-range discount."""
    df_anomalies = pd.DataFrame(
        {
            "order_id": ["O-1", "O-1", "O-2", "O-3"],
            "customer_id": ["C-1", "C-1", "C-2", "C-3"],
            "order_date": pd.to_datetime(
                ["2016-01-01", "2016-01-01", "2016-01-05", "2016-01-10"]
            ),
            "ship_date": pd.to_datetime(
                ["2016-01-04", "2016-01-04", "2016-01-02", "2016-01-12"]
            ),  # Row 2 invalid ship date
            "sales": [100.0, 100.0, -50.0, 200.0],  # Rows 0 & 1 duplicate, Row 2 negative sales
            "quantity": [2, 2, 0, 4],  # Row 2 invalid quantity
            "discount": [0.1, 0.1, 1.5, 0.2],  # Row 2 invalid discount > 1.0
            "profit": [20.0, 20.0, -10.0, 40.0],
            "extra_null": [1.0, 1.0, 3.0, None],  # Row 3 contains null
        }
    )

    report = validate_data_integrity(df_anomalies, raise_on_error=False)
    assert report["duplicate_rows"] == 1
    assert report["total_missing_values"] == 1
    assert report["invalid_quantities_count"] == 1
    assert report["negative_sales_count"] == 1
    assert report["invalid_discounts_count"] == 1
    assert report["invalid_shipping_dates_count"] == 1

    with pytest.raises(ValueError, match="Data integrity validation failed"):
        validate_data_integrity(df_anomalies, raise_on_error=True)
