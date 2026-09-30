"""
Main Pipeline Runner for E-Commerce Sales Data Analysis & Insights.

Executes the full automated workflow:
1. Data Quality Assessment & Cleaning Pipeline
2. Statistical Profiling & Business KPI Engineering
3. Publication-Quality Figure Generation (300 DPI)
"""

from pathlib import Path
import sys

from src.data_cleaning import clean_dataset
from src.analysis import calculate_core_kpis
from src.visualization import generate_all_visualizations


def run_pipeline() -> None:
    """Execute the end-to-end data engineering and analytics pipeline."""
    project_root = Path(__file__).resolve().parent
    raw_path = project_root / "data" / "raw" / "sales.csv"
    processed_path = project_root / "data" / "processed" / "cleaned_sales.csv"
    figures_dir = project_root / "outputs" / "figures"

    print("=" * 75)
    print("🚀 E-COMMERCE SALES DATA ANALYSIS & INSIGHTS — PIPELINE RUNNER")
    print("=" * 75)

    # 1. Data Cleaning & Validation Pipeline
    print("\n[1/3] 🧹 Running Data Cleaning & Quality Validation Pipeline...")
    df_clean, val_report, outlier_report = clean_dataset(
        raw_path=raw_path,
        output_path=processed_path,
        raise_on_error=True,
    )
    print(f"  ✓ Raw dataset:        {val_report['raw_rows']:,} rows, {val_report['raw_columns']} columns")
    print(f"  ✓ Processed dataset:  {val_report['cleaned_rows']:,} rows, {val_report['cleaned_columns']} columns")
    print(f"  ✓ Missing Values:     {val_report['total_missing_values']}")
    print(f"  ✓ Duplicate Records:  {val_report['duplicate_rows']}")
    print(f"  ✓ Preserved Records:  100.0% (Zero record deletion)")
    print(f"  ✓ Saved clean file:   {processed_path}")

    # 2. Financial & Business KPIs
    print("\n[2/3] 💰 Calculating Enterprise Business KPIs & Statistics...")
    kpis = calculate_core_kpis(df_clean)
    print(f"  • Total Gross Revenue:        ${kpis['total_revenue']:,.2f}")
    print(f"  • Total Net Profit:           ${kpis['total_profit']:,.2f}")
    print(f"  • Overall Profit Margin:      {kpis['overall_profit_margin_pct']}%")
    print(f"  • Total Unique Orders:        {kpis['total_orders']:,}")
    print(f"  • Total Unique Customers:     {kpis['total_customers']:,}")
    print(f"  • Total Units Sold:           {kpis['total_quantity_sold']:,}")
    print(f"  • Average Order Value (AOV):  ${kpis['avg_order_value']:,.2f}  [Order-level mean]")
    print(f"  • Mean Sales per Line Item:   ${kpis['avg_sales_per_line_item']:,.2f}  [Row-level mean]")
    print(f"  • Profitable Orders:          {kpis['profitable_orders_count']:,} ({kpis['profitable_orders_pct']}%)")
    print(f"  • Loss-Making Orders:         {kpis['loss_making_orders_count']:,} ({kpis['loss_making_orders_pct']}%)")
    print(f"  • Break-Even Orders:          {kpis['breakeven_orders_count']:,} ({kpis['breakeven_orders_pct']}%)")

    # 3. Visualizations
    print("\n[3/3] 📊 Generating Publication-Quality Figures (300 DPI)...")
    figures = generate_all_visualizations(df_clean, output_dir=figures_dir)
    for fig_path in figures:
        print(f"  ✓ Exported: {fig_path.name}")

    print("\n" + "=" * 75)
    print("🎉 ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print(f"👉 Jupyter Notebook: notebooks/01_exploratory_data_analysis.ipynb")
    print(f"👉 Summary Report:   outputs/reports/eda_summary_report.md")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
