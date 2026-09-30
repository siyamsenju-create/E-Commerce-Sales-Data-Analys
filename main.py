"""
Main Entrypoint Script for E-Commerce Sales Data Analysis & Insights.
Executes the full pipeline: Data Cleaning -> KPI Analysis -> Visualization Generation.
"""

import os
import sys
import pandas as pd

from src.data_cleaning import clean_dataset
from src.analysis import calculate_core_kpis, calculate_distribution_statistics, calculate_category_metrics
from src.visualization import generate_all_visualizations


def main():
    print("=" * 70)
    print("🚀 E-COMMERCE SALES DATA ANALYSIS & INSIGHTS — PIPELINE RUNNER")
    print("=" * 70)

    # 1. Data Cleaning & Validation Pipeline
    print("\n[1/3] 🧹 Running Data Cleaning & Quality Validation Pipeline...")
    raw_path = "data/raw/sales.csv"
    processed_path = "data/processed/cleaned_sales.csv"
    
    df_clean, val_report, outlier_report = clean_dataset(
        raw_path=raw_path,
        output_path=processed_path
    )
    print(f"  ✓ Raw dataset:       {val_report['raw_rows']:,} rows, {val_report['raw_columns']} columns")
    print(f"  ✓ Processed dataset: {val_report['cleaned_rows']:,} rows, {val_report['cleaned_columns']} columns")
    print(f"  ✓ Total Null Values: {val_report['total_missing_values']}")
    print(f"  ✓ Saved clean file:  {processed_path}")

    # 2. Financial & Business KPIs
    print("\n[2/3] 💰 Calculating Enterprise Business KPIs & Statistics...")
    kpis = calculate_core_kpis(df_clean)
    print(f"  • Total Gross Revenue:     ${kpis['total_revenue']:,.2f}")
    print(f"  • Total Net Profit:        ${kpis['total_profit']:,.2f}")
    print(f"  • Overall Profit Margin:   {kpis['overall_profit_margin_pct']}%")
    print(f"  • Total Unique Orders:     {kpis['total_orders']:,}")
    print(f"  • Total Unique Customers:  {kpis['total_customers']:,}")
    print(f"  • Average Order Value:     ${kpis['avg_order_value']:,.2f}")
    print(f"  • Profitable Orders:       {kpis['profitable_orders_count']:,} ({kpis['profitable_orders_pct']}%)")
    print(f"  • Loss-Making Orders:      {kpis['loss_making_orders_count']:,} ({kpis['loss_making_orders_pct']}%)")

    # 3. Visualizations
    print("\n[3/3] 📊 Generating Publication-Quality Figures (300 DPI)...")
    output_dir = "outputs/figures"
    figures = generate_all_visualizations(df_clean, output_dir=output_dir)
    for fig_path in figures:
        print(f"  ✓ Exported: {fig_path}")

    print("\n" + "=" * 70)
    print("🎉 ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print(f"👉 Jupyter Notebook: notebooks/01_exploratory_data_analysis.ipynb")
    print(f"👉 Summary Report:   outputs/reports/eda_summary_report.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
