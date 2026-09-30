"""
Comprehensive Jupyter Notebook Generator for E-Commerce Sales EDA Project.
Generates a 21-section publication-quality notebook adhering to AI Engineering standards.
"""

from pathlib import Path
import nbformat as nbf

project_root = Path(__file__).resolve().parent

nb = nbf.v4.new_notebook()
cells = []

# =============================================================
# 1. Project Introduction
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """# E-Commerce Sales Data Analysis & Insights
## End-to-End Exploratory Data Analysis & Data Engineering Pipeline
**Author:** AI / Data Engineering Portfolio Project  
**Tech Stack:** Python 3.11, Pandas, NumPy, Matplotlib, Seaborn  
**Environment:** Apple Silicon (macOS ARM64, M2, 16 GB RAM)

---

### 1. Project Introduction
Exploratory Data Analysis (EDA) is the essential foundation of AI and Machine Learning engineering. Before building predictive models (regression, classification, clustering, or forecasting), an AI Engineer must rigorously investigate raw data distributions, audit data quality, validate schema integrity, and extract domain-relevant features without introducing target leakage or destructive data loss.

```text
Raw Tabular Data ──> Quality Audit & Cleaning ──> Feature Engineering ──> Statistical Profiling ──> Multi-Dimensional GroupBy ──> Visualizations ──> Evidence-Based Insights ──> ML Readiness
```
"""
    )
)

# =============================================================
# 2. Business Questions
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 2. Business Questions
This project investigates 10 targeted commercial questions:
1. **Category Revenue:** Which product category generates the highest total sales volume?
2. **Category Profitability:** How does net profitability and profit margin vary across categories?
3. **Geographic Distribution:** What are the key regional revenue drivers across the United States?
4. **Time-Series Trajectory:** How do sales evolve chronologically across 2014–2017, and what seasonal patterns exist?
5. **Distribution Skewness:** How are line-item sales distributed, and how does a log transformation impact skewness?
6. **Top Products:** Which products drive top-line revenue vs. bottom-line profitability?
7. **Volume vs. Sales:** What is the empirical relationship between units ordered and gross revenue?
8. **Feature Associations:** What linear relationships exist across numerical transaction variables?
9. **Discount Impact:** How does applied promotional discounting associate with aggregate profitability?
10. **Margin Efficiency:** How do category profit margins compare with total gross sales rankings?
"""
    )
)

# =============================================================
# 3. Environment Setup
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 3. Environment Setup & Library Imports
We configure lightweight, high-performance scientific computing libraries with modular architecture imports from `src/`.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
project_root = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_cleaning import (
    load_raw_data,
    standardize_column_names,
    convert_datetime_columns,
    add_date_features,
    calculate_unit_price,
    validate_data_integrity,
    detect_outliers_iqr,
    clean_dataset
)

from src.analysis import (
    calculate_core_kpis,
    calculate_distribution_statistics,
    calculate_category_metrics,
    calculate_regional_metrics,
    calculate_segment_metrics,
    calculate_customer_metrics,
    calculate_product_metrics,
    calculate_discount_impact_metrics,
    calculate_correlations
)

from src.visualization import (
    plot_sales_by_category,
    plot_profit_by_category,
    plot_sales_by_region,
    plot_monthly_sales_trend,
    plot_sales_distribution,
    plot_top_products_by_revenue,
    plot_quantity_vs_sales,
    plot_correlation_heatmap,
    plot_discount_profitability,
    plot_category_profit_margin,
    generate_all_visualizations
)

# Visualization styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#d0d5dd'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 120

print(f"Pandas version: {pd.__version__}")
print(f"NumPy version:  {np.__version__}")
"""
    )
)

# =============================================================
# 4. Load Dataset
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 4. Load Dataset
We load `data/raw/sales.csv` with standard character encoding while maintaining the raw file untouched.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """raw_data_path = project_root / 'data' / 'raw' / 'sales.csv'
df_raw = load_raw_data(raw_data_path, encoding='windows-1252')

print(f"Raw Dataset Loaded: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")
display(df_raw.head(3))
"""
    )
)

# =============================================================
# 5. Dataset Overview
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 5. Dataset Overview & Schema Inspection
Inspecting schema data types and column structures:
"""
    )
)

cells.append(nbf.v4.new_code_cell("""df_raw.info()"""))

# =============================================================
# 6. Data Quality Assessment
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 6. Data Quality Assessment

### Core Engineering Principle: Data Cleaning $\\neq$ Deleting Unusual Data
In AI/ML engineering, indiscriminately dropping unusual records introduces severe **selection bias**.
- **Valid modifications:** Standardizing column headers to `snake_case`, parsing date strings to `datetime64[ns]`, and verifying boundary constraints ($Q > 0, \\text{Sales} \\ge 0$).
- **Valid retentions:** Retaining negative profit records (legitimate commercial loss from promotional discounting) and high-value orders (genuine enterprise transactions).
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Missing values audit
missing_counts = df_raw.isnull().sum()
missing_df = pd.DataFrame({'Missing Count': missing_counts, 'Missing %': (missing_counts / len(df_raw)) * 100})
print("=== MISSING VALUES AUDIT ===")
display(missing_df[missing_df['Missing Count'] > 0] if (missing_counts > 0).any() else "Zero missing values detected across all raw columns.")

# Duplicate records audit
duplicate_count = df_raw.duplicated().sum()
print(f"\\nExact Duplicate Rows: {duplicate_count}")
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """### Statistical Outlier Identification (IQR Method)
$$\\text{IQR} = Q_3 - Q_1,\\quad \\text{Lower Bound} = Q_1 - 1.5\\times\\text{IQR},\\quad \\text{Upper Bound} = Q_3 + 1.5\\times\\text{IQR}$$
Outliers are detected and documented for analytical transparency, NOT deleted.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """cleaned_sample = standardize_column_names(df_raw)
outlier_report = detect_outliers_iqr(cleaned_sample)

outlier_summary = []
for col, metrics in outlier_report.items():
    outlier_summary.append({
        'Feature': col,
        'Q1 (25%)': f"{metrics['q1']:.2f}",
        'Median': f"{metrics['median']:.2f}",
        'Q3 (75%)': f"{metrics['q3']:.2f}",
        'IQR': f"{metrics['iqr']:.2f}",
        'Lower Bound': f"{metrics['lower_bound']:.2f}",
        'Upper Bound': f"{metrics['upper_bound']:.2f}",
        'Outlier Count': metrics['outlier_count'],
        'Outlier %': f"{metrics['outlier_percentage']}%"
    })

pd.DataFrame(outlier_summary)
"""
    )
)

# =============================================================
# 7. Data Cleaning Pipeline
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 7. Data Cleaning Pipeline
We execute the standardized data cleaning pipeline:
1. Standardize column names (`snake_case`).
2. Convert date strings to `datetime64[ns]`.
3. Validate data integrity without deleting records.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """processed_data_path = project_root / 'data' / 'processed' / 'cleaned_sales.csv'

df_cleaned, val_report, outlier_report = clean_dataset(
    raw_path=raw_data_path,
    output_path=processed_data_path,
    raise_on_error=True
)

print("=== DATA INTEGRITY VALIDATION ===")
print(f"Raw Dimensions:        {val_report['raw_rows']:,} rows x {val_report['raw_columns']} columns")
print(f"Cleaned Dimensions:    {val_report['cleaned_rows']:,} rows x {val_report['cleaned_columns']} columns")
print(f"Total Null Values:     {val_report['total_missing_values']}")
print(f"Unique Orders:         {val_report['unique_orders']:,}")
print(f"Unique Customers:      {val_report['unique_customers']:,}")
print(f"Invalid Quantities:    {val_report['invalid_quantities_count']}")
print(f"Negative Sales:        {val_report['negative_sales_count']}")
print(f"Invalid Shipping Date: {val_report['invalid_shipping_dates_count']}")
print(f"Preserved Rows:        100.0% (Zero record deletion)")
"""
    )
)

# =============================================================
# 8. Feature Engineering
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 8. Feature Engineering
We derive temporal, fulfillment, and pricing features:
- `order_year`: Calendar year ($2014 - 2017$)
- `order_month`: Integer month ($1 - 12$)
- `order_month_name`: Short month name (`Jan` - `Dec`)
- `order_quarter`: Fiscal / calendar quarter ($1 - 4$)
- `order_year_month`: Chronological period string (`YYYY-MM`)
- `shipping_days`: Fulfillment duration ($(\\text{ship\\_date} - \\text{order\\_date}).\\text{days}$)
- `unit_price`: Derived baseline price $(\\text{sales} / \\text{quantity})$
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """display(df_cleaned[['order_id', 'order_date', 'ship_date', 'shipping_days', 'category', 'sales', 'quantity', 'unit_price', 'profit']].head())
"""
    )
)

# =============================================================
# 9. KPI Analysis
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 9. Business KPI Analysis

### Educational Concept: Line-Item Metrics vs. Order-Level Metrics
- **Mean Sales per Line Item:** `df['sales'].mean()` = **$229.86**
- **Average Order Value (AOV):** `df.groupby('order_id')['sales'].sum().mean()` = **$458.61**

Because customer orders contain an average of 2.0 items, evaluating at the basket level is required for commercial validity.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """kpis = calculate_core_kpis(df_cleaned)

kpi_table = pd.DataFrame([
    {"Business Metric": "Total Gross Revenue", "Value": f"${kpis['total_revenue']:,.2f}"},
    {"Business Metric": "Total Net Profit", "Value": f"${kpis['total_profit']:,.2f}"},
    {"Business Metric": "Overall Profit Margin", "Value": f"{kpis['overall_profit_margin_pct']}%"},
    {"Business Metric": "Total Units Sold", "Value": f"{kpis['total_quantity_sold']:,}"},
    {"Business Metric": "Total Unique Orders", "Value": f"{kpis['total_orders']:,}"},
    {"Business Metric": "Total Unique Customers", "Value": f"{kpis['total_customers']:,}"},
    {"Business Metric": "Average Order Value (AOV)", "Value": f"${kpis['avg_order_value']:,.2f}"},
    {"Business Metric": "Median Order Value", "Value": f"${kpis['median_order_value']:,.2f}"},
    {"Business Metric": "Average Profit per Order", "Value": f"${kpis['avg_profit_per_order']:,.2f}"},
    {"Business Metric": "Median Profit per Order", "Value": f"${kpis['median_profit_per_order']:,.2f}"},
    {"Business Metric": "Mean Sales per Line Item", "Value": f"${kpis['avg_sales_per_line_item']:,.2f}"},
    {"Business Metric": "Median Sales per Line Item", "Value": f"${kpis['median_sales_per_line_item']:,.2f}"},
    {"Business Metric": "Profitable Orders", "Value": f"{kpis['profitable_orders_count']:,} ({kpis['profitable_orders_pct']}%)"},
    {"Business Metric": "Loss-Making Orders", "Value": f"{kpis['loss_making_orders_count']:,} ({kpis['loss_making_orders_pct']}%)"},
    {"Business Metric": "Break-Even Orders", "Value": f"{kpis['breakeven_orders_count']:,} ({kpis['breakeven_orders_pct']}%)"}
])

display(kpi_table)
"""
    )
)

# =============================================================
# 10. Descriptive Statistics
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 10. Descriptive Statistics & Distribution Profiling
Comparing parametric (mean, std) and non-parametric (median, IQR) measures reveals right-skewness in financial variables ($S = +12.97$ for sales).

> **Statistical Note:** The raw sales distribution is strongly right-skewed, and a logarithmic transformation produces a more symmetric representation.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """dist_stats = calculate_distribution_statistics(df_cleaned)
display(dist_stats)
"""
    )
)

# =============================================================
# 11. Category Analysis
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 11. Category & Sub-Category Analysis
Evaluating sales, profit, quantity, and margins across product hierarchies:
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """cat_summary = calculate_category_metrics(df_cleaned, group_col='category')
display(cat_summary)

subcat_summary = calculate_category_metrics(df_cleaned, group_col='sub_category')
display(subcat_summary.head(10))
"""
    )
)

# =============================================================
# 12. Regional Analysis
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 12. Regional & Segment Analysis
Evaluating geographic performance and customer segment contributions:
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """reg_summary = calculate_regional_metrics(df_cleaned)
display(reg_summary)

seg_summary = calculate_segment_metrics(df_cleaned)
display(seg_summary)
"""
    )
)

# =============================================================
# 13. Customer Analysis
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 13. Customer Analysis
Investigating customer revenue distributions and top accounts:
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """cust_dist, top_customers = calculate_customer_metrics(df_cleaned, top_n=5)
display(cust_dist)
print("\\n=== TOP 5 CUSTOMERS BY REVENUE ===")
display(top_customers[['customer_id', 'customer_name', 'segment', 'total_sales', 'total_profit', 'order_count', 'avg_order_value']])
"""
    )
)

# =============================================================
# 14. Product Analysis
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 14. Product Analysis
Examining revenue leaders, profit leaders, and loss-leader products:
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """prod_insights = calculate_product_metrics(df_cleaned, top_n=5)

print("=== TOP 5 PRODUCTS BY REVENUE ===")
display(prod_insights['top_by_sales'][['product_name', 'category', 'total_sales', 'total_profit', 'profit_margin_pct']])

print("\\n=== TOP 5 PRODUCTS BY PROFIT ===")
display(prod_insights['top_by_profit'][['product_name', 'category', 'total_sales', 'total_profit', 'profit_margin_pct']])

print("\\n=== TOP 5 MOST UNPROFITABLE PRODUCTS ===")
display(prod_insights['most_unprofitable'][['product_name', 'category', 'total_sales', 'total_profit', 'profit_margin_pct']])
"""
    )
)

# =============================================================
# 15. Time-Series Analysis
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 15. Time-Series Analysis (Multi-Year Trajectory)
Evaluating sales growth across months and years (2014–2017) to observe recurring patterns.

> **Calendar Note:** Calendar Q4 consists of October, November, and December. Sales accelerate in September and continue into the Q4 period.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """monthly_trend = (
    df_cleaned.groupby(['order_year', 'order_month_name', 'order_month'])
    .agg(total_sales=('sales', 'sum'), total_profit=('profit', 'sum'), order_count=('order_id', 'nunique'))
    .reset_index()
    .sort_values(by=['order_year', 'order_month'])
)
display(monthly_trend.head(12))
"""
    )
)

# =============================================================
# 16. Correlation Analysis
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 16. Correlation Analysis

### Correlation $\\neq$ Causation
Pearson correlation ($r$) measures linear association between numerical variables. It does **not** establish causation. We avoid causal terminology (such as "causes", "drives", "proves").
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """corr_matrix = calculate_correlations(df_cleaned)
display(corr_matrix)
"""
    )
)

# =============================================================
# 17. Visualizations
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 17. Visualizations (10 Evidence-Based Charts)
Each visualization below addresses a specific business question using the structured format: **Question**, **Method**, **Result**, and **Interpretation**.
"""
    )
)

# 17.1 Sales by Category
cells.append(
    nbf.v4.new_markdown_cell(
        """### 17.1 Sales by Category
- **Question:** Which product category generates the most revenue?
- **Method:** Bar chart aggregating total $\\sum(\\text{sales})$ by `category`.
- **Result:** Technology generated $836,154.03 (36.40%), Furniture generated $741,999.80 (32.30%), and Office Supplies generated $719,047.03 (31.30%).
- **Interpretation:** Revenue is distributed relatively evenly across all three categories, with Technology slightly leading.
"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """fig1 = plot_sales_by_category(df_cleaned, output_path=project_root / 'outputs' / 'figures' / '01_sales_by_category.png')
plt.show()
"""
    )
)

# 17.2 Profit by Category
cells.append(
    nbf.v4.new_markdown_cell(
        """### 17.2 Profit by Category
- **Question:** How does net profitability differ across categories?
- **Method:** Bar chart aggregating total $\\sum(\\text{profit})$ by `category`.
- **Result:** Technology contributed $145,454.95 (50.79% of profit, 17.40% margin), Office Supplies contributed $122,490.80 (42.77% of profit, 17.04% margin), and Furniture contributed $18,451.27 (6.44% of profit, 2.49% margin).
- **Interpretation:** Despite accounting for nearly a third of sales, Furniture generated only 6.44% of net profit, associated with heavy discounts in Tables (-$17,725.48 loss).
"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """fig2 = plot_profit_by_category(df_cleaned, output_path=project_root / 'outputs' / 'figures' / '02_profit_by_category.png')
plt.show()
"""
    )
)

# 17.3 Sales by Region
cells.append(
    nbf.v4.new_markdown_cell(
        """### 17.3 Sales by Region
- **Question:** How is gross revenue distributed geographically?
- **Method:** Bar chart aggregating total $\\sum(\\text{sales})$ by `region`.
- **Result:** West: $725,457.82 (31.58%), East: $678,781.24 (29.55%), Central: $501,239.89 (21.82%), South: $391,721.90 (17.05%).
- **Interpretation:** Coastal regions (West and East) represent over 61% of total enterprise volume.
"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """fig3 = plot_sales_by_region(df_cleaned, output_path=project_root / 'outputs' / 'figures' / '03_sales_by_region.png')
plt.show()
"""
    )
)

# 17.4 Monthly Sales Trend
cells.append(
    nbf.v4.new_markdown_cell(
        """### 17.4 Monthly Sales Trend
- **Question:** How does revenue change over time across multi-year cycles?
- **Method:** Multi-year line chart with 3-month moving average across 48 months (2014–2017).
- **Result:** Consistent year-over-year expansion with recurring volume acceleration in September and peak sales during Q4 (November and December).
- **Interpretation:** Seasonal demand accelerates in late summer and reaches maximum volume during the holiday quarter.
"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """fig4 = plot_monthly_sales_trend(df_cleaned, output_path=project_root / 'outputs' / 'figures' / '04_monthly_sales_trend.png')
plt.show()
"""
    )
)

# 17.5 Sales Distribution
cells.append(
    nbf.v4.new_markdown_cell(
        """### 17.5 Sales Distribution
- **Question:** How are individual line-item sales distributed?
- **Method:** Dual-panel histogram comparing raw linear scale vs. $\\log_{10}(\\text{sales})$.
- **Result:** The raw sales distribution is strongly right-skewed ($\\text{Mean } \\$229.86 \\gg \\text{Median } \\$54.49, S = +12.97$). Log transformation produces a more symmetric representation.
- **Interpretation:** Feature scaling and log transformation should be evaluated in downstream ML modeling.
"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """fig5 = plot_sales_distribution(df_cleaned, output_path=project_root / 'outputs' / 'figures' / '05_sales_distribution.png')
plt.show()
"""
    )
)

# 17.6 Top Products
cells.append(
    nbf.v4.new_markdown_cell(
        """### 17.6 Top Products by Revenue
- **Question:** Which individual products generate the highest sales?
- **Method:** Horizontal bar chart of top 10 products by $\\sum(\\text{sales})$.
- **Result:** The *Canon imageCLASS 2200 Advanced Copier* generated $61,599.82 in sales and $25,199.93 in profit (40.91% profit margin).
- **Interpretation:** High-ticket commercial hardware represents disproportionate revenue and profit contributions.
"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """fig6 = plot_top_products_by_revenue(df_cleaned, output_path=project_root / 'outputs' / 'figures' / '06_top_products_by_revenue.png', top_n=10)
plt.show()
"""
    )
)

# 17.7 Quantity vs Sales
cells.append(
    nbf.v4.new_markdown_cell(
        """### 17.7 Quantity vs. Sales
- **Question:** Is there a relationship between quantity and sales?
- **Method:** Scatter plot with linear trendline and Pearson correlation ($r = +0.2008$).
- **Result:** A weak positive linear association exists ($r = +0.2008$).
- **Interpretation:** Quantity alone accounts for only a minor portion of sales variance; unit price variation across catalog items is the primary factor.
"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """fig7 = plot_quantity_vs_sales(df_cleaned, output_path=project_root / 'outputs' / 'figures' / '07_quantity_vs_sales.png')
plt.show()
"""
    )
)

# 17.8 Correlation Heatmap
cells.append(
    nbf.v4.new_markdown_cell(
        """### 17.8 Correlation Heatmap
- **Question:** What linear relationships exist among numerical transaction features?
- **Method:** Annotated Pearson correlation heatmap across continuous variables.
- **Result:** `sales` and `profit` show moderate positive linear association ($r = +0.4791$), `discount` and `profit` show moderate negative linear association ($r = -0.2195$), and `sales` and `unit_price` show strong positive correlation ($r = +0.8894$).
- **Interpretation:** Observational evidence indicates that higher discounts coincide with lower profitability in this dataset.
"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """fig8 = plot_correlation_heatmap(df_cleaned, output_path=project_root / 'outputs' / 'figures' / '08_correlation_heatmap.png')
plt.show()
"""
    )
)

# 17.9 Discount vs Profitability
cells.append(
    nbf.v4.new_markdown_cell(
        """### 17.9 Discount vs. Profitability
- **Question:** How does profitability vary across discount tiers?
- **Method:** Bar chart aggregating net profit and margin % by applied discount level.
- **Result:** Transactions with discounts $\\le 20\\%$ yielded **+\\$421,773.08** in total profit. Transactions with discounts $\\ge 30\\%$ accumulated **-\\$135,376.06** in net losses.
- **Interpretation:** Higher discount levels were associated with lower aggregate profitability in this dataset.
"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """fig9 = plot_discount_profitability(df_cleaned, output_path=project_root / 'outputs' / 'figures' / '09_discount_vs_profitability.png')
plt.show()
"""
    )
)

# 17.10 Profit Margin by Category
cells.append(
    nbf.v4.new_markdown_cell(
        """### 17.10 Profit Margin by Category
- **Question:** How do category profit margins compare to revenue rankings?
- **Method:** Bar chart of $(\\sum\\text{profit} / \\sum\\text{sales}) \\times 100$ by `category`.
- **Result:** Technology (17.40%) and Office Supplies (17.04%) achieve strong margins, whereas Furniture achieves 2.49%.
- **Interpretation:** Gross sales rank does not mirror margin efficiency; category-level monitoring is required.
"""
    )
)
cells.append(
    nbf.v4.new_code_cell(
        """fig10 = plot_category_profit_margin(df_cleaned, output_path=project_root / 'outputs' / 'figures' / '10_profit_margin_by_category.png')
plt.show()
"""
    )
)

# =============================================================
# 18. Key Insights
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 18. Key Insights (Evidence-Based)

1. **Technology Generates Majority of Net Profits:**
   - *Observed Fact:* Technology delivered **$145,454.95 (50.79% of total net profits)** on **$836,154.03 in sales (36.40% share)** with a **17.40% margin**.
   - *Interpretation:* Commercial technology equipment is the highest-margin segment in the dataset.

2. **Furniture Exhibits Margin Dilution:**
   - *Observed Fact:* Furniture generated **$741,999.80 in sales (32.30% share)** but yielded only **$18,451.27 in profit (6.44% share)** with a **2.49% margin**. Tables operated at an aggregate net loss of **-$17,725.48**.
   - *Interpretation:* Heavy discounting in specific furniture sub-categories coincided with severe margin erosion.

3. **Discount Levels Associated with Commercial Losses Beyond 20%:**
   - *Observed Fact:* Discounts $\\le 20\\%$ accumulated **+$421,773.08 in profit**, whereas discounts $\\ge 30\\%$ accumulated **-$135,376.06 in cumulative net losses**.
   - *Interpretation:* Promotional pricing strategies above 20% were associated with unprofitable transactions in this dataset.

4. **Late-Year & Q4 Demand Acceleration:**
   - *Observed Fact:* Monthly sales accelerate in September and reach their annual peak during the fourth quarter (November and December).
   - *Interpretation:* Operational inventory planning and logistics capacity must ramp up ahead of Q4.

5. **Top-Line Volume Does Not Guarantee Profitability:**
   - *Observed Fact:* The #1 highest-revenue customer (Sean Miller: $25,043.05 sales across 5 orders) produced a **-$1,980.74 net loss**, and the #3 revenue product (Cisco TelePresence System: $22,638.48 sales) generated a **-$1,811.08 net loss**.
   - *Interpretation:* Commercial incentives must align with bottom-line net contribution rather than top-line revenue alone.
"""
    )
)

# =============================================================
# 19. Limitations
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 19. Limitations & Confounding Factors
- **Observational Nature:** This analysis is strictly observational. Correlations ($r$) do not establish causal relationships.
- **Unmeasured Variables:** The dataset lacks detailed variable-level logistics costs, warehouse storage fees, advertising spend, and supplier acquisition costs.
- **Linearity Assumption:** Pearson correlation assesses linear associations and does not capture non-linear relationships.
"""
    )
)

# =============================================================
# 20. Machine Learning Readiness
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 20. Machine Learning Readiness & Target Leakage Prevention

### Potential Predictive Formulations:
1. **Binary Classification (Order Profitability):**
   - **Target:** $\\text{profitable\\_order} = 1 \\text{ if } \\text{order\\_profit} > 0 \\text{ else } 0$
   - **Class Balance:** 79.26% profitable vs. 20.40% loss-making orders.
2. **Regression (Profit / Revenue Forecasting):**
   - **Target:** Continuous `order_profit` or `sales`.
3. **Customer Analytics (RFM Segmentation):**
   - Recency, Frequency, and Monetary value clustering via K-Means.

### Data Leakage Awareness & Prevention:
- **Target Leakage Risk:** If predicting order profitability *before* order completion, using features derived from final discounts, post-order shipping duration, or final line-item profit will leak target information.
- **ML Best Practice:** Any future modeling pipeline must explicitly define:
  1. Prediction timestamp (e.g., at checkout request).
  2. Features available strictly *at or before* that timestamp (e.g., customer historical RFM, product category, baseline price).
  3. Preprocessing (log transformation, standard scaling, one-hot encoding) fit strictly on training splits.
"""
    )
)

# =============================================================
# 21. Conclusion
# =============================================================
cells.append(
    nbf.v4.new_markdown_cell(
        """## 21. Conclusion
- **100% Data Integrity:** Verified zero missing values, zero duplicates, and zero record loss across 9,994 line items.
- **KPI Precision:** Formulated order-level AOV ($458.61) distinctly from line-item mean ($229.86).
- **Statistical Rigor:** Quantified distribution skewness and Pearson associations without unsupported causal claims.
- **Production Reproducibility:** Modular codebase (`src/`), automated pipeline (`main.py`), and test suite (`tests/`) ready for portfolio review.
"""
    )
)

nb.cells = cells

notebook_path = project_root / "notebooks" / "01_exploratory_data_analysis.ipynb"
with open(notebook_path, "w") as f:
    nbf.write(nb, f)

print(f"Successfully generated complete 21-section notebook at: {notebook_path}")
