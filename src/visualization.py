"""
Visualization Module for E-Commerce Sales Data Analysis.

This module provides modular, reusable functions using Matplotlib and Seaborn
to generate publication-quality, evidence-based figures answering specific
analytical business questions.
"""

from pathlib import Path
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Configure consistent styling aesthetics
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = "Helvetica"
plt.rcParams["axes.edgecolor"] = "#d0d5dd"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#f2f4f7"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

PALETTE_PRIMARY = "#2563eb"  # Royal blue
PALETTE_SECONDARY = "#0d9488"  # Teal
PALETTE_ACCENT = "#f59e0b"  # Amber
PALETTE_DANGER = "#ef4444"  # Red
PALETTE_DARK = "#1e293b"  # Slate dark


def format_currency_k(value: float) -> str:
    """Format numeric float to compact USD ($k / $M)."""
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"${value / 1_000:.1f}k"
    return f"${value:.2f}"


def plot_sales_by_category(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = "outputs/figures/01_sales_by_category.png",
) -> plt.Figure:
    """
    Chart 1: Total Revenue by Product Category.
    Question: Which product category generates the most revenue?
    """
    cat_sales = (
        df.groupby("category")["sales"]
        .sum()
        .reset_index()
        .sort_values(by="sales", ascending=False)
    )
    total_sales = cat_sales["sales"].sum()
    cat_sales["share_pct"] = (cat_sales["sales"] / total_sales) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        cat_sales["category"],
        cat_sales["sales"],
        color=["#2563eb", "#3b82f6", "#60a5fa"],
        edgecolor="#1e3a8a",
        width=0.55,
    )

    for bar, pct in zip(bars, cat_sales["share_pct"]):
        y_val = bar.get_height()
        ax.annotate(
            f"{format_currency_k(y_val)}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, y_val),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=PALETTE_DARK,
        )

    ax.set_title(
        "Total Revenue by Product Category", fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Product Category", fontsize=11, labelpad=8)
    ax.set_ylabel("Total Gross Sales (USD)", fontsize=11, labelpad=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency_k(x)))
    ax.set_ylim(0, cat_sales["sales"].max() * 1.18)

    plt.tight_layout()
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300)
    return fig


def plot_profit_by_category(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = "outputs/figures/02_profit_by_category.png",
) -> plt.Figure:
    """
    Chart 2: Total Profit by Product Category.
    Question: How does profitability differ across categories?
    """
    cat_profit = (
        df.groupby("category")["profit"]
        .sum()
        .reset_index()
        .sort_values(by="profit", ascending=False)
    )
    total_profit = cat_profit["profit"].sum()
    cat_profit["share_pct"] = (cat_profit["profit"] / total_profit) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        cat_profit["category"],
        cat_profit["profit"],
        color=["#0d9488", "#14b8a6", "#f59e0b"],
        edgecolor="#0f766e",
        width=0.55,
    )

    for bar, pct in zip(bars, cat_profit["share_pct"]):
        y_val = bar.get_height()
        ax.annotate(
            f"{format_currency_k(y_val)}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, y_val),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=PALETTE_DARK,
        )

    ax.set_title(
        "Total Net Profit by Product Category", fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Product Category", fontsize=11, labelpad=8)
    ax.set_ylabel("Total Net Profit (USD)", fontsize=11, labelpad=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency_k(x)))
    ax.set_ylim(0, cat_profit["profit"].max() * 1.18)

    plt.tight_layout()
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300)
    return fig


def plot_sales_by_region(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = "outputs/figures/03_sales_by_region.png",
) -> plt.Figure:
    """
    Chart 3: Regional Sales Distribution.
    Question: How is revenue distributed geographically?
    """
    reg_sales = (
        df.groupby("region")["sales"]
        .sum()
        .reset_index()
        .sort_values(by="sales", ascending=False)
    )
    total_sales = reg_sales["sales"].sum()
    reg_sales["share_pct"] = (reg_sales["sales"] / total_sales) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        reg_sales["region"],
        reg_sales["sales"],
        color="#475569",
        edgecolor="#1e293b",
        width=0.55,
    )

    for bar, pct in zip(bars, reg_sales["share_pct"]):
        y_val = bar.get_height()
        ax.annotate(
            f"{format_currency_k(y_val)}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, y_val),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=PALETTE_DARK,
        )

    ax.set_title(
        "Total Revenue by Geographic Region", fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Geographic Region", fontsize=11, labelpad=8)
    ax.set_ylabel("Total Gross Sales (USD)", fontsize=11, labelpad=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency_k(x)))
    ax.set_ylim(0, reg_sales["sales"].max() * 1.18)

    plt.tight_layout()
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300)
    return fig


def plot_monthly_sales_trend(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = "outputs/figures/04_monthly_sales_trend.png",
) -> plt.Figure:
    """
    Chart 4: Chronological Monthly Sales Trend (Multi-Year Trajectory).
    Question: How does revenue change over time?
    """
    monthly = (
        df.groupby(["order_year_month", "order_year", "order_month"])["sales"]
        .sum()
        .reset_index()
        .sort_values(by=["order_year", "order_month"])
    )

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(
        range(len(monthly)),
        monthly["sales"],
        marker="o",
        markersize=5,
        color=PALETTE_PRIMARY,
        linewidth=2,
        label="Monthly Revenue",
    )

    # 3-month rolling average
    rolling_3m = monthly["sales"].rolling(window=3, min_periods=1).mean()
    ax.plot(
        range(len(monthly)),
        rolling_3m,
        color=PALETTE_ACCENT,
        linewidth=2.2,
        linestyle="--",
        label="3-Month Moving Average",
    )

    tick_indices = list(range(0, len(monthly), 3))
    tick_labels = [monthly.iloc[i]["order_year_month"] for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)

    ax.set_title(
        "Chronological Monthly Sales Trend (2014 – 2017)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Order Year-Month", fontsize=11, labelpad=8)
    ax.set_ylabel("Monthly Sales (USD)", fontsize=11, labelpad=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency_k(x)))
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300)
    return fig


def plot_sales_distribution(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = "outputs/figures/05_sales_distribution.png",
) -> plt.Figure:
    """
    Chart 5: Sales Distribution (Dual-View: Raw Linear Scale & Log10 Scale).
    Question: How are individual line-item sales distributed?
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Raw linear distribution
    sns.histplot(
        df["sales"],
        bins=60,
        kde=False,
        color=PALETTE_PRIMARY,
        ax=ax1,
        edgecolor="#1e3a8a",
    )
    mean_val = df["sales"].mean()
    median_val = df["sales"].median()

    ax1.axvline(
        mean_val,
        color=PALETTE_DANGER,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: ${mean_val:.2f}",
    )
    ax1.axvline(
        median_val,
        color=PALETTE_SECONDARY,
        linestyle="-",
        linewidth=1.5,
        label=f"Median: ${median_val:.2f}",
    )
    ax1.set_title(
        "Line-Item Sales Distribution (Linear Scale)", fontsize=11, fontweight="bold"
    )
    ax1.set_xlabel("Sales (USD)", fontsize=10)
    ax1.set_ylabel("Frequency (Order Lines)", fontsize=10)
    ax1.set_xlim(0, 3000)
    ax1.legend(loc="upper right")

    # Panel 2: Log10 transformed distribution
    log_sales = np.log10(df["sales"])
    sns.histplot(
        log_sales,
        bins=40,
        kde=True,
        color=PALETTE_SECONDARY,
        ax=ax2,
        edgecolor="#0f766e",
    )
    ax2.set_title("Log10 Transformed Sales Distribution", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Log10(Sales)", fontsize=10)
    ax2.set_ylabel("Density / Frequency", fontsize=10)

    plt.suptitle(
        "Line-Item Sales Distribution & Skewness Analysis",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300)
    return fig


def plot_top_products_by_revenue(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = "outputs/figures/06_top_products_by_revenue.png",
    top_n: int = 10,
) -> plt.Figure:
    """
    Chart 6: Top 10 Products by Gross Revenue.
    Question: Which products generate the highest sales?
    """
    top_prods = (
        df.groupby(["product_name", "category"])["sales"]
        .sum()
        .reset_index()
        .sort_values(by="sales", ascending=True)
        .tail(top_n)
    )

    top_prods["display_name"] = top_prods["product_name"].apply(
        lambda x: x[:42] + "..." if len(x) > 45 else x
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        top_prods["display_name"],
        top_prods["sales"],
        color="#3b82f6",
        edgecolor="#1d4ed8",
        height=0.65,
    )

    for bar in bars:
        w_val = bar.get_width()
        ax.annotate(
            f" {format_currency_k(w_val)}",
            xy=(w_val, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=PALETTE_DARK,
        )

    ax.set_title(
        f"Top {top_n} Products by Revenue", fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Total Gross Revenue (USD)", fontsize=11, labelpad=8)
    ax.set_ylabel("Product Name", fontsize=11, labelpad=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency_k(x)))
    ax.set_xlim(0, top_prods["sales"].max() * 1.18)

    plt.tight_layout()
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300)
    return fig


def plot_quantity_vs_sales(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = "outputs/figures/07_quantity_vs_sales.png",
) -> plt.Figure:
    """
    Chart 7: Scatter Plot of Order Line Quantity vs. Sales.
    Question: Is there a relationship between quantity and sales?
    """
    corr_val = float(df["quantity"].corr(df["sales"]))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(
        df["quantity"],
        df["sales"],
        alpha=0.35,
        color=PALETTE_PRIMARY,
        edgecolors="none",
        s=25,
    )

    z = np.polyfit(df["quantity"], df["sales"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df["quantity"].min(), df["quantity"].max(), 100)
    ax.plot(
        x_range,
        p(x_range),
        color=PALETTE_DANGER,
        linestyle="--",
        linewidth=2,
        label=f"Linear Trendline (Pearson r = +{corr_val:.2f})",
    )

    ax.set_title(
        "Order-Line Quantity vs. Gross Sales", fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Quantity (Units per Line Item)", fontsize=11, labelpad=8)
    ax.set_ylabel("Sales (USD)", fontsize=11, labelpad=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency_k(x)))
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = "outputs/figures/08_correlation_heatmap.png",
    numeric_columns: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Chart 8: Pearson Correlation Heatmap for Numerical Features.
    Question: What linear relationships exist among numerical variables?
    """
    if numeric_columns is None:
        numeric_columns = [
            col
            for col in ["sales", "quantity", "discount", "profit", "unit_price"]
            if col in df.columns
        ]

    corr_matrix = df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.6,
        vmax=1.0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"},
        ax=ax,
    )

    ax.set_title(
        "Correlation Heatmap of Numerical Features",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300)
    return fig


def plot_discount_profitability(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = "outputs/figures/09_discount_vs_profitability.png",
) -> plt.Figure:
    """
    Chart 9: Aggregate Profitability by Discount Level.
    Question: How does profitability vary across discount levels?
    """
    disc_summary = (
        df.groupby("discount")
        .agg(
            total_profit=("profit", "sum"),
            total_sales=("sales", "sum"),
            order_count=("sales", "count"),
        )
        .reset_index()
    )
    disc_summary["profit_margin_pct"] = (
        disc_summary["total_profit"] / disc_summary["total_sales"]
    ) * 100

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [
        PALETTE_SECONDARY if p >= 0 else PALETTE_DANGER
        for p in disc_summary["total_profit"]
    ]

    bars = ax.bar(
        [f"{int(d*100)}%" for d in disc_summary["discount"]],
        disc_summary["total_profit"],
        color=colors,
        edgecolor=PALETTE_DARK,
        width=0.6,
    )

    ax.axhline(0, color=PALETTE_DARK, linewidth=1)

    for bar, margin in zip(bars, disc_summary["profit_margin_pct"]):
        y_val = bar.get_height()
        offset = 5 if y_val >= 0 else -15
        ax.annotate(
            f"{format_currency_k(y_val)}\n({margin:.0f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, y_val),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if y_val >= 0 else "top",
            fontsize=8.5,
            fontweight="bold",
            color=PALETTE_DARK,
        )

    ax.set_title(
        "Aggregate Net Profit & Margin by Discount Tier",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Applied Discount Level (%)", fontsize=11, labelpad=8)
    ax.set_ylabel("Total Net Profit (USD)", fontsize=11, labelpad=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency_k(x)))

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min * 1.25, y_max * 1.25)

    plt.tight_layout()
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300)
    return fig


def plot_category_profit_margin(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = "outputs/figures/10_profit_margin_by_category.png",
) -> plt.Figure:
    """
    Chart 10: Profit Margin Percentage by Product Category.
    Question: How do category profit margins compare to revenue rankings?
    """
    cat_margin = (
        df.groupby("category")
        .agg(
            total_sales=("sales", "sum"),
            total_profit=("profit", "sum"),
        )
        .reset_index()
    )
    cat_margin["profit_margin_pct"] = (
        cat_margin["total_profit"] / cat_margin["total_sales"]
    ) * 100
    cat_margin = cat_margin.sort_values(by="profit_margin_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        cat_margin["category"],
        cat_margin["profit_margin_pct"],
        color=["#0d9488", "#14b8a6", "#f59e0b"],
        edgecolor="#0f766e",
        width=0.55,
    )

    for bar in bars:
        y_val = bar.get_height()
        ax.annotate(
            f"{y_val:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, y_val),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=PALETTE_DARK,
        )

    ax.set_title(
        "Net Profit Margin (%) by Product Category",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Product Category", fontsize=11, labelpad=8)
    ax.set_ylabel("Profit Margin (%)", fontsize=11, labelpad=8)
    ax.set_ylim(0, cat_margin["profit_margin_pct"].max() * 1.25)

    plt.tight_layout()
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=300)
    return fig


def generate_all_visualizations(
    df: pd.DataFrame,
    output_dir: Union[str, Path] = "outputs/figures",
) -> List[Path]:
    """
    Generate and save all 10 project visualizations at 300 DPI.

    Args:
        df: Cleaned DataFrame.
        output_dir: Directory path where PNG charts will be saved.

    Returns:
        List[Path]: List of Path objects to saved PNG figures.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[Path] = []

    plots = [
        ("01_sales_by_category.png", plot_sales_by_category),
        ("02_profit_by_category.png", plot_profit_by_category),
        ("03_sales_by_region.png", plot_sales_by_region),
        ("04_monthly_sales_trend.png", plot_monthly_sales_trend),
        ("05_sales_distribution.png", plot_sales_distribution),
        ("06_top_products_by_revenue.png", plot_top_products_by_revenue),
        ("07_quantity_vs_sales.png", plot_quantity_vs_sales),
        ("08_correlation_heatmap.png", plot_correlation_heatmap),
        ("09_discount_vs_profitability.png", plot_discount_profitability),
        ("10_profit_margin_by_category.png", plot_category_profit_margin),
    ]

    for fname, func in plots:
        fpath = out_dir / fname
        func(df, output_path=fpath)
        plt.close("all")
        generated_files.append(fpath)

    return generated_files
