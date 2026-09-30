"""
Statistical Analysis & Business Metrics Module for E-Commerce Sales Data.

This module provides modular, reusable functions for calculating enterprise business
KPIs, distribution statistics (mean, median, std, variance, IQR, skewness),
multi-dimensional aggregations (category, region, customer segment, product),
and correlation matrices with strict differentiation between correlation and causation.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def calculate_core_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate high-level enterprise business KPIs.

    Crucial Distinction:
    - Line-item metrics: Computed across all individual line items (n=9,994).
    - Order-level metrics: Computed across unique order_id transactions (n=5,009).

    AOV (Average Order Value) is an order-level metric:
        1. Group by order_id.
        2. Sum sales per order.
        3. Compute the mean across orders ($458.61).

    Args:
        df: Cleaned DataFrame.

    Returns:
        Dict[str, Any]: Dictionary containing core KPIs and order-level metrics.
    """
    if df.empty:
        return {
            "total_revenue": 0.0,
            "total_profit": 0.0,
            "overall_profit_margin_pct": 0.0,
            "total_quantity_sold": 0,
            "total_orders": 0,
            "total_customers": 0,
            "total_line_items": 0,
            "avg_order_value": 0.0,
            "median_order_value": 0.0,
            "avg_profit_per_order": 0.0,
            "median_profit_per_order": 0.0,
            "avg_items_per_order": 0.0,
            "avg_line_items_per_order": 0.0,
            "avg_sales_per_line_item": 0.0,
            "median_sales_per_line_item": 0.0,
            "avg_profit_per_line_item": 0.0,
            "median_profit_per_line_item": 0.0,
            "profitable_orders_count": 0,
            "profitable_orders_pct": 0.0,
            "loss_making_orders_count": 0,
            "loss_making_orders_pct": 0.0,
            "breakeven_orders_count": 0,
            "breakeven_orders_pct": 0.0,
        }

    total_revenue = float(df["sales"].sum())
    total_profit = float(df["profit"].sum())
    total_quantity = int(df["quantity"].sum())
    total_line_items = len(df)
    total_orders = int(df["order_id"].nunique())
    total_customers = int(df["customer_id"].nunique())

    # Order-level aggregations
    order_grouped = df.groupby("order_id").agg(
        order_sales=("sales", "sum"),
        order_profit=("profit", "sum"),
        order_items=("quantity", "sum"),
        line_item_count=("sales", "count"),
    )

    avg_order_value = float(order_grouped["order_sales"].mean())
    median_order_value = float(order_grouped["order_sales"].median())
    avg_profit_per_order = float(order_grouped["order_profit"].mean())
    median_profit_per_order = float(order_grouped["order_profit"].median())
    avg_items_per_order = float(order_grouped["order_items"].mean())
    avg_line_items_per_order = float(order_grouped["line_item_count"].mean())

    # Profitability metrics
    overall_profit_margin = (
        (total_profit / total_revenue) * 100 if total_revenue > 0 else 0.0
    )

    profitable_orders_count = int((order_grouped["order_profit"] > 0).sum())
    loss_orders_count = int((order_grouped["order_profit"] < 0).sum())
    breakeven_orders_count = int((order_grouped["order_profit"] == 0).sum())

    profitable_orders_pct = (
        (profitable_orders_count / total_orders) * 100 if total_orders > 0 else 0.0
    )
    loss_orders_pct = (
        (loss_orders_count / total_orders) * 100 if total_orders > 0 else 0.0
    )
    breakeven_orders_pct = (
        (breakeven_orders_count / total_orders) * 100 if total_orders > 0 else 0.0
    )

    # Line-item level metrics for comparison
    avg_sales_per_line_item = float(df["sales"].mean())
    median_sales_per_line_item = float(df["sales"].median())
    avg_profit_per_line_item = float(df["profit"].mean())
    median_profit_per_line_item = float(df["profit"].median())

    return {
        "total_revenue": round(total_revenue, 2),
        "total_profit": round(total_profit, 2),
        "overall_profit_margin_pct": round(overall_profit_margin, 2),
        "total_quantity_sold": total_quantity,
        "total_orders": total_orders,
        "total_customers": total_customers,
        "total_line_items": total_line_items,
        "avg_order_value": round(avg_order_value, 2),
        "median_order_value": round(median_order_value, 2),
        "avg_profit_per_order": round(avg_profit_per_order, 2),
        "median_profit_per_order": round(median_profit_per_order, 2),
        "avg_items_per_order": round(avg_items_per_order, 2),
        "avg_line_items_per_order": round(avg_line_items_per_order, 2),
        "avg_sales_per_line_item": round(avg_sales_per_line_item, 2),
        "median_sales_per_line_item": round(median_sales_per_line_item, 2),
        "avg_profit_per_line_item": round(avg_profit_per_line_item, 2),
        "median_profit_per_line_item": round(median_profit_per_line_item, 2),
        "profitable_orders_count": profitable_orders_count,
        "profitable_orders_pct": round(profitable_orders_pct, 2),
        "loss_making_orders_count": loss_orders_count,
        "loss_making_orders_pct": round(loss_orders_pct, 2),
        "breakeven_orders_count": breakeven_orders_count,
        "breakeven_orders_pct": round(breakeven_orders_pct, 2),
    }


def calculate_distribution_statistics(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Calculate parametric and non-parametric distribution statistics.

    Metrics included:
    - Parametric: Mean, Standard Deviation, Variance
    - Non-Parametric: Min, 25% (Q1), Median (50%), 75% (Q3), Max, IQR
    - Shape: Skewness and descriptive characterization

    Args:
        df: Input DataFrame.
        numeric_columns: List of columns to evaluate. Defaults to
            ['sales', 'profit', 'quantity', 'unit_price', 'discount', 'shipping_days'].

    Returns:
        pd.DataFrame: Comprehensive summary table indexed by feature.
    """
    if numeric_columns is None:
        numeric_columns = [
            col
            for col in [
                "sales",
                "profit",
                "quantity",
                "unit_price",
                "discount",
                "shipping_days",
            ]
            if col in df.columns
        ]

    stats_list = []
    for col in numeric_columns:
        if col not in df.columns or df[col].empty:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue

        q1 = float(series.quantile(0.25))
        median = float(series.median())
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        mean = float(series.mean())
        std = float(series.std()) if len(series) > 1 else 0.0
        var = float(series.var()) if len(series) > 1 else 0.0
        skewness = float(series.skew()) if len(series) > 2 else 0.0

        if abs(skewness) < 0.5:
            shape_desc = "Fairly Symmetric"
        elif skewness >= 2.0:
            shape_desc = "Strongly Right-Skewed (log transformation produces more symmetric representation)"
        elif skewness >= 0.5:
            shape_desc = "Moderately Right-Skewed"
        elif skewness <= -2.0:
            shape_desc = "Strongly Left-Skewed"
        else:
            shape_desc = "Moderately Left-Skewed"

        stats_list.append(
            {
                "Feature": col,
                "Count": len(series),
                "Mean": round(mean, 2),
                "Median": round(median, 2),
                "Std Dev": round(std, 2),
                "Variance": round(var, 2),
                "Min": round(float(series.min()), 2),
                "Q1 (25%)": round(q1, 2),
                "Q3 (75%)": round(q3, 2),
                "IQR": round(iqr, 2),
                "Max": round(float(series.max()), 2),
                "Skewness": round(skewness, 2),
                "Distribution Shape": shape_desc,
            }
        )

    return pd.DataFrame(stats_list)


def calculate_category_metrics(
    df: pd.DataFrame, group_col: str = "category"
) -> pd.DataFrame:
    """
    Aggregate financial and volume KPIs across product categories or sub-categories.

    Args:
        df: Input DataFrame.
        group_col: Column to group by ('category' or 'sub_category').

    Returns:
        pd.DataFrame: Summary table sorted descending by total sales.
    """
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()

    total_sales_all = df["sales"].sum()
    total_profit_all = df["profit"].sum()

    summary = (
        df.groupby(group_col)
        .agg(
            total_sales=("sales", "sum"),
            total_profit=("profit", "sum"),
            total_quantity=("quantity", "sum"),
            order_line_count=("sales", "count"),
            unique_orders=("order_id", "nunique"),
            avg_unit_price=("unit_price", "mean"),
        )
        .reset_index()
    )

    summary["profit_margin_pct"] = (
        (summary["total_profit"] / summary["total_sales"]) * 100
    ).round(2)
    summary["sales_share_pct"] = (
        (summary["total_sales"] / total_sales_all) * 100
        if total_sales_all > 0
        else 0.0
    ).round(2)
    summary["profit_share_pct"] = (
        (summary["total_profit"] / total_profit_all) * 100
        if total_profit_all > 0
        else 0.0
    ).round(2)

    summary["total_sales"] = summary["total_sales"].round(2)
    summary["total_profit"] = summary["total_profit"].round(2)
    summary["avg_unit_price"] = summary["avg_unit_price"].round(2)

    return summary.sort_values(by="total_sales", ascending=False).reset_index(drop=True)


def calculate_regional_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sales, profit, margins, and customer reach by geographic region.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Regional performance table sorted by total sales.
    """
    if df.empty or "region" not in df.columns:
        return pd.DataFrame()

    total_sales_all = df["sales"].sum()
    total_profit_all = df["profit"].sum()

    regional = (
        df.groupby("region")
        .agg(
            total_sales=("sales", "sum"),
            total_profit=("profit", "sum"),
            total_quantity=("quantity", "sum"),
            unique_orders=("order_id", "nunique"),
            unique_customers=("customer_id", "nunique"),
        )
        .reset_index()
    )

    regional["profit_margin_pct"] = (
        (regional["total_profit"] / regional["total_sales"]) * 100
    ).round(2)
    regional["sales_share_pct"] = (
        (regional["total_sales"] / total_sales_all) * 100
        if total_sales_all > 0
        else 0.0
    ).round(2)
    regional["profit_share_pct"] = (
        (regional["total_profit"] / total_profit_all) * 100
        if total_profit_all > 0
        else 0.0
    ).round(2)

    regional["total_sales"] = regional["total_sales"].round(2)
    regional["total_profit"] = regional["total_profit"].round(2)

    return regional.sort_values(by="total_sales", ascending=False).reset_index(drop=True)


def calculate_segment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate performance across customer segments (Consumer, Corporate, Home Office).

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Customer segment metrics table.
    """
    if df.empty or "segment" not in df.columns:
        return pd.DataFrame()

    total_sales_all = df["sales"].sum()

    seg = (
        df.groupby("segment")
        .agg(
            total_sales=("sales", "sum"),
            total_profit=("profit", "sum"),
            total_quantity=("quantity", "sum"),
            unique_orders=("order_id", "nunique"),
            unique_customers=("customer_id", "nunique"),
        )
        .reset_index()
    )

    seg["profit_margin_pct"] = (
        (seg["total_profit"] / seg["total_sales"]) * 100
    ).round(2)
    seg["avg_revenue_per_order"] = (
        seg["total_sales"] / seg["unique_orders"]
    ).round(2)
    seg["avg_revenue_per_customer"] = (
        seg["total_sales"] / seg["unique_customers"]
    ).round(2)
    seg["sales_share_pct"] = (
        (seg["total_sales"] / total_sales_all) * 100
        if total_sales_all > 0
        else 0.0
    ).round(2)

    seg["total_sales"] = seg["total_sales"].round(2)
    seg["total_profit"] = seg["total_profit"].round(2)

    return seg.sort_values(by="total_sales", ascending=False).reset_index(drop=True)


def calculate_customer_metrics(
    df: pd.DataFrame, top_n: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze customer-level revenue, profit, and purchasing behavior.

    Args:
        df: Input DataFrame.
        top_n: Number of highest-revenue customers to extract.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            1. Overall customer summary distribution.
            2. Top N highest-revenue customer table.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    cust_grouped = (
        df.groupby(["customer_id", "customer_name", "segment"])
        .agg(
            total_sales=("sales", "sum"),
            total_profit=("profit", "sum"),
            total_quantity=("quantity", "sum"),
            order_count=("order_id", "nunique"),
            line_item_count=("sales", "count"),
        )
        .reset_index()
    )

    cust_grouped["profit_margin_pct"] = (
        (cust_grouped["total_profit"] / cust_grouped["total_sales"]) * 100
    ).round(2)
    cust_grouped["avg_order_value"] = (
        cust_grouped["total_sales"] / cust_grouped["order_count"]
    ).round(2)

    top_customers = (
        cust_grouped.sort_values(by="total_sales", ascending=False)
        .head(top_n)
        .copy()
        .reset_index(drop=True)
    )
    top_customers["total_sales"] = top_customers["total_sales"].round(2)
    top_customers["total_profit"] = top_customers["total_profit"].round(2)

    cust_dist = pd.DataFrame(
        [
            {
                "Metric": "Total Unique Customers",
                "Value": f"{len(cust_grouped):,}",
            },
            {
                "Metric": "Mean Revenue per Customer",
                "Value": f"${cust_grouped['total_sales'].mean():,.2f}",
            },
            {
                "Metric": "Median Revenue per Customer",
                "Value": f"${cust_grouped['total_sales'].median():,.2f}",
            },
            {
                "Metric": "Mean Profit per Customer",
                "Value": f"${cust_grouped['total_profit'].mean():,.2f}",
            },
            {
                "Metric": "Mean Orders per Customer",
                "Value": f"{cust_grouped['order_count'].mean():.2f}",
            },
            {
                "Metric": "Max Orders by a Single Customer",
                "Value": f"{cust_grouped['order_count'].max()}",
            },
        ]
    )

    return cust_dist, top_customers


def calculate_product_metrics(
    df: pd.DataFrame, top_n: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Analyze product performance across revenue, profit, quantity, and loss-leaders.

    Dimensions:
    1. Top products by sales (revenue leaders)
    2. Top products by profit (margin leaders)
    3. Top products by quantity sold (volume leaders)
    4. Most unprofitable products (loss leaders)

    Args:
        df: Input DataFrame.
        top_n: Number of records to return per ranking.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing top/bottom product tables.
    """
    if df.empty:
        empty_df = pd.DataFrame()
        return {
            "top_by_sales": empty_df,
            "top_by_profit": empty_df,
            "top_by_quantity": empty_df,
            "most_unprofitable": empty_df,
        }

    prod_grouped = (
        df.groupby(["product_id", "product_name", "category", "sub_category"])
        .agg(
            total_sales=("sales", "sum"),
            total_profit=("profit", "sum"),
            total_quantity=("quantity", "sum"),
            order_line_count=("sales", "count"),
        )
        .reset_index()
    )

    prod_grouped["profit_margin_pct"] = (
        (prod_grouped["total_profit"] / prod_grouped["total_sales"]) * 100
    ).round(2)
    prod_grouped["total_sales"] = prod_grouped["total_sales"].round(2)
    prod_grouped["total_profit"] = prod_grouped["total_profit"].round(2)

    top_by_sales = (
        prod_grouped.sort_values(by="total_sales", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    top_by_profit = (
        prod_grouped.sort_values(by="total_profit", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    top_by_quantity = (
        prod_grouped.sort_values(by="total_quantity", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    most_unprofitable = (
        prod_grouped.sort_values(by="total_profit", ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )

    return {
        "top_by_sales": top_by_sales,
        "top_by_profit": top_by_profit,
        "top_by_quantity": top_by_quantity,
        "most_unprofitable": most_unprofitable,
    }


def calculate_discount_impact_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate financial behavior across discrete discount tiers.

    Note:
        This observational analysis describes associations present in the dataset.
        It does not establish that discounts alone caused lower profit.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Table of revenue, profit, margin, and order count by discount tier.
    """
    if df.empty or "discount" not in df.columns:
        return pd.DataFrame()

    discount_perf = (
        df.groupby("discount")
        .agg(
            transaction_count=("sales", "count"),
            total_sales=("sales", "sum"),
            total_profit=("profit", "sum"),
            avg_profit=("profit", "mean"),
            avg_sales=("sales", "mean"),
        )
        .reset_index()
    )

    discount_perf["profit_margin_pct"] = (
        (discount_perf["total_profit"] / discount_perf["total_sales"]) * 100
    ).round(2)
    discount_perf["total_sales"] = discount_perf["total_sales"].round(2)
    discount_perf["total_profit"] = discount_perf["total_profit"].round(2)
    discount_perf["avg_profit"] = discount_perf["avg_profit"].round(2)
    discount_perf["avg_sales"] = discount_perf["avg_sales"].round(2)

    return discount_perf.sort_values(by="discount").reset_index(drop=True)


def calculate_correlations(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Calculate Pearson correlation matrix for numerical features.

    Note:
        Pearson correlation measures linear association and does NOT establish causation.

    Args:
        df: Input DataFrame.
        numeric_columns: List of numerical columns. Defaults to
            ['sales', 'quantity', 'discount', 'profit', 'unit_price'].

    Returns:
        pd.DataFrame: Correlation matrix rounded to 4 decimal places.
    """
    if df.empty:
        return pd.DataFrame()

    if numeric_columns is None:
        numeric_columns = [
            col
            for col in ["sales", "quantity", "discount", "profit", "unit_price"]
            if col in df.columns
        ]

    return df[numeric_columns].corr().round(4)
