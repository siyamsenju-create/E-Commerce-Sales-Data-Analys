"""
Unit tests for business KPI analysis and statistical modeling in src/analysis.py.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.analysis import (
    calculate_category_metrics,
    calculate_core_kpis,
    calculate_correlations,
    calculate_customer_metrics,
    calculate_discount_impact_metrics,
    calculate_distribution_statistics,
    calculate_product_metrics,
    calculate_regional_metrics,
    calculate_segment_metrics,
)


@pytest.fixture
def project_cleaned_df() -> pd.DataFrame:
    """Load the full project cleaned dataset for numerical KPI assertion."""
    cleaned_path = Path("data/processed/cleaned_sales.csv")
    if not cleaned_path.exists():
        pytest.skip("Cleaned dataset not found; run cleaning first")
    return pd.read_csv(cleaned_path)


def test_core_kpi_calculations(project_cleaned_df: pd.DataFrame):
    """Verify exact enterprise KPIs against benchmark ground truth."""
    kpis = calculate_core_kpis(project_cleaned_df)

    # Enterprise Financials
    assert kpis["total_revenue"] == pytest.approx(2297200.86, rel=1e-4)
    assert kpis["total_profit"] == pytest.approx(286397.02, rel=1e-4)
    assert kpis["overall_profit_margin_pct"] == pytest.approx(12.47, abs=0.05)

    # Entity counts
    assert kpis["total_line_items"] == 9994
    assert kpis["total_orders"] == 5009
    assert kpis["total_customers"] == 793
    assert kpis["total_quantity_sold"] == 37873

    # Order-level vs line-item distinctions
    assert kpis["avg_order_value"] == pytest.approx(458.61, abs=0.05)
    assert kpis["median_order_value"] == pytest.approx(151.96, abs=0.05)
    assert kpis["avg_profit_per_order"] == pytest.approx(57.18, abs=0.05)
    assert kpis["median_profit_per_order"] == pytest.approx(16.28, abs=0.05)
    assert kpis["avg_sales_per_line_item"] == pytest.approx(229.86, abs=0.05)

    # Order outcome breakdown
    assert kpis["profitable_orders_count"] == 3970
    assert kpis["loss_making_orders_count"] == 1022
    assert kpis["breakeven_orders_count"] == 17


def test_category_metrics(project_cleaned_df: pd.DataFrame):
    """Verify category totals and shares."""
    cat_df = calculate_category_metrics(project_cleaned_df, group_col="category")
    tech = cat_df[cat_df["category"] == "Technology"].iloc[0]
    office = cat_df[cat_df["category"] == "Office Supplies"].iloc[0]
    furn = cat_df[cat_df["category"] == "Furniture"].iloc[0]

    # Technology
    assert tech["total_sales"] == pytest.approx(836154.03, abs=0.1)
    assert tech["total_profit"] == pytest.approx(145454.95, abs=0.1)
    assert tech["profit_margin_pct"] == pytest.approx(17.40, abs=0.1)
    assert tech["sales_share_pct"] == pytest.approx(36.40, abs=0.1)
    assert tech["profit_share_pct"] == pytest.approx(50.79, abs=0.1)

    # Office Supplies
    assert office["total_sales"] == pytest.approx(719047.03, abs=0.1)
    assert office["total_profit"] == pytest.approx(122490.80, abs=0.1)
    assert office["profit_margin_pct"] == pytest.approx(17.04, abs=0.1)

    # Furniture
    assert furn["total_sales"] == pytest.approx(741999.80, abs=0.1)
    assert furn["total_profit"] == pytest.approx(18451.27, abs=0.1)
    assert furn["profit_margin_pct"] == pytest.approx(2.49, abs=0.1)


def test_regional_metrics(project_cleaned_df: pd.DataFrame):
    """Verify regional revenue distributions."""
    reg_df = calculate_regional_metrics(project_cleaned_df)
    west = reg_df[reg_df["region"] == "West"].iloc[0]
    east = reg_df[reg_df["region"] == "East"].iloc[0]
    central = reg_df[reg_df["region"] == "Central"].iloc[0]
    south = reg_df[reg_df["region"] == "South"].iloc[0]

    assert west["total_sales"] == pytest.approx(725457.82, abs=0.1)
    assert west["sales_share_pct"] == pytest.approx(31.58, abs=0.1)

    assert east["total_sales"] == pytest.approx(678781.24, abs=0.1)
    assert east["sales_share_pct"] == pytest.approx(29.55, abs=0.1)

    assert central["total_sales"] == pytest.approx(501239.89, abs=0.1)
    assert central["sales_share_pct"] == pytest.approx(21.82, abs=0.1)

    assert south["total_sales"] == pytest.approx(391721.90, abs=0.1)
    assert south["sales_share_pct"] == pytest.approx(17.05, abs=0.1)


def test_top_customer_metrics(project_cleaned_df: pd.DataFrame):
    """Verify top customer identification and loss profile."""
    _, top_cust = calculate_customer_metrics(project_cleaned_df, top_n=5)
    top_1 = top_cust.iloc[0]

    assert top_1["customer_name"] == "Sean Miller"
    assert top_1["total_sales"] == pytest.approx(25043.05, abs=0.1)
    assert top_1["total_profit"] == pytest.approx(-1980.74, abs=0.1)
    assert top_1["order_count"] == 5


def test_top_product_metrics(project_cleaned_df: pd.DataFrame):
    """Verify top product revenue and profit characteristics."""
    prod_metrics = calculate_product_metrics(project_cleaned_df, top_n=5)
    top_prod = prod_metrics["top_by_sales"].iloc[0]

    assert "Canon imageCLASS 2200 Advanced Copier" in top_prod["product_name"]
    assert top_prod["total_sales"] == pytest.approx(61599.82, abs=0.1)
    assert top_prod["total_profit"] == pytest.approx(25199.93, abs=0.1)
    assert top_prod["profit_margin_pct"] == pytest.approx(40.91, abs=0.1)


def test_correlations_matrix(project_cleaned_df: pd.DataFrame):
    """Verify correlation values and matrix structure."""
    corr = calculate_correlations(project_cleaned_df)

    assert corr.loc["sales", "quantity"] == pytest.approx(0.2008, abs=0.01)
    assert corr.loc["sales", "profit"] == pytest.approx(0.4791, abs=0.01)
    assert corr.loc["discount", "profit"] == pytest.approx(-0.2195, abs=0.01)
    assert corr.loc["sales", "unit_price"] == pytest.approx(0.8894, abs=0.01)
