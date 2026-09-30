"""
Comprehensive Jupyter Notebook Generator for E-Commerce Sales EDA Project.
Generates a 16-section notebook adhering strictly to AI Engineering portfolio standards.
"""

import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# =============================================================
# 1. Project Introduction
# =============================================================
cells.append(nbf.v4.new_markdown_cell("""# E-Commerce Sales Data Analysis & Insights
## End-to-End Exploratory Data Analysis (EDA) Pipeline
**Author:** AI Engineering Portfolio Project  
**Tech Stack:** Python 3.11, Pandas, NumPy, Matplotlib, Seaborn  
**Environment:** Apple Silicon (macOS ARM64)

---

### 1. Project Introduction
Exploratory Data Analysis (EDA) is the foundational cornerstone of AI and Machine Learning engineering. Before any predictive model (regression, classification, clustering, or deep learning) can be architected, an AI Engineer must rigorously investigate raw data distributions, audit data quality, validate schema integrity, and extract domain-relevant features.

```text
Raw Tabular Data ──> Quality Audit & Cleaning ──> Statistical Profiling ──> EDA & GroupBy ──> Visualizations ──> Evidence-Based Insights ──> ML Preparedness
```

#### Analytical Scope & Objectives:
1. **Data Quality & Integrity:** Validate schemas, handle missing values and duplicates without blind record deletion, verify chronological logic, and detect statistical outliers (IQR).
2. **Financial KPI Engineering:** Accurately differentiate line-item metrics from order-level metrics ($AOV$).
3. **Multi-Dimensional GroupBy:** Assess category, sub-category, geographic region, customer segments, and individual products.
4. **Time-Series Analysis:** Uncover long-term growth trajectories and seasonality.
5. **Correlation vs. Causation:** Statistically evaluate linear relationships (Pearson $r$) and study the impact of discounting on net margins.
6. **Actionable Insights:** Formulate data-driven strategic findings backed by measurable figures.
"""))

# =============================================================
# 2. Import Libraries
# =============================================================
cells.append(nbf.v4.new_markdown_cell("""## 2. Import Libraries & Configure Environment
We utilize only lightweight, standard scientific computing and visualization libraries optimized for macOS ARM64.
"""))

cells.append(nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path to import modular functions
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

# Configure visualization styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#d0d5dd'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 120

print(f"Pandas version: {pd.__version__}")
print(f"NumPy version:  {np.__version__}")
"""))

# =============================================================
# 3. Load Dataset
# =============================================================
cells.append(nbf.v4.new_markdown_cell("""## 3. Load Dataset
We load the raw CSV from `data/raw/sales.csv` using `load_raw_data()` while keeping the raw file untouched.
"""))

cells.append(nbf.v4.new_code_cell("""raw_data_path = os.path.join(project_root, 'data', 'raw', 'sales.csv')
df_raw = load_raw_data(raw_data_path, encoding='windows-1252')

print(f"Raw Dataset Shape: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")
display(df_raw.head(3))
"""))

# =============================================================
# 4. Dataset Overview
# =============================================================
cells.append(nbf.v4.new_markdown_cell("""## 4. Dataset Overview
Inspecting dataset schema, column types, and non-null counts:
"""))

cells.append(nbf.v4.new_code_cell("""df_raw.info()"""))

# =============================================================
# 5. Data Quality Assessment
# =============================================================
cells.append(nbf.v4.new_markdown_cell("""## 5. Data Quality Assessment

### Core Rule: Data Cleaning $\\neq$ Deleting Unusual Data
In AI/ML engineering, indiscriminately dropping unusual records introduces severe **selection bias**.
- **Valid modifications:** Standardizing column headers to `snake_case`, parsing date strings to `datetime64[ns]`, and verifying boundary constraints ($Q > 0, \\text{Sales} \\ge 0$).
- **Valid retentions:** Retaining negative profit records (legitimate commercial loss from deep discounting) and high-value orders (genuine enterprise transactions).
"""))

cells.append(nbf.v4.new_code_cell("""# Missing values audit
missing_counts = df_raw.isnull().sum()
missing_df = pd.DataFrame({'Missing Count': missing_counts, 'Missing %': (missing_counts / len(df_raw)) * 100})
print("=== MISSING VALUES AUDIT ===")
display(missing_df[missing_df['Missing Count'] > 0] if (missing_counts > 0).any() else "Zero missing values detected across all raw columns.")

# Duplicate records audit
duplicate_count = df_raw.duplicated().sum()
print(f"\\nExact Duplicate Rows: {duplicate_count}")
"""))

# =============================================================
# 6. Data Cleaning
# =============================================================
cells.append(nbf.v4.new_markdown_cell("""## 6. Data Cleaning Pipeline & Feature Engineering
We standardize headers, convert dates, engineer date attributes (`order_year`, `order_month`, `order_quarter`), derive `unit_price = sales / quantity`, and export to `data/processed/cleaned_sales.csv`.
"""))

cells.append(nbf.v4.new_code_cell("""processed_data_path = os.path.join(project_root, 'data', 'processed', 'cleaned_sales.csv')

df_cleaned, val_report, outlier_report = clean_dataset(
    raw_path=raw_data_path,
    output_path=processed_data_path
)

print("=== DATA INTEGRITY VALIDATION ===")
print(f"Raw Dimensions:     {val_report['raw_rows']:,} rows x {val_report['raw_columns']} columns")
print(f"Cleaned Dimensions: {val_report['cleaned_rows']:,} rows x {val_report['cleaned_columns']} columns")
print(f"Total Null Values:  {val_report['total_missing_values']}")
print(f"Unique Orders:      {val_report['unique_orders']:,}")
print(f"Unique Customers:   {val_report['unique_customers']:,}")
print(f"Invalid Quantities: {val_report['invalid_quantities_count']}")
print(f"Negative Sales:     {val_report['negative_sales_count']}")
print(f"Invalid Shipping:   {val_report['invalid_shipping_dates_count']}")

display(df_cleaned[['order_id', 'order_date', 'customer_name', 'category', 'sub_category', 'sales', 'quantity', 'unit_price', 'profit']].head())
"""))

cells.append(nbf.v4.new_markdown_cell("""### Statistical Outlier Identification (IQR Method)
$$\\text{IQR} = Q_3 - Q_1,\\quad \\text{Lower} = Q_1 - 1.5\\times\\text{IQR},\\quad \\text{Upper} = Q_3 + 1.5\\times\\text{IQR}$$
"""))

cells.append(nbf.v4.new_code_cell("""outlier_summary = []
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
"""))

# =============================================================
# 7. Descriptive Statistics
# =============================================================
cells.append(nbf.v4.new_markdown_cell("""## 7. Descriptive Statistics & Business KPIs

### Educational Concept: Order-Line Level vs. Order-Level Aggregation
The dataset consists of **9,994 line items** representing **5,009 unique customer checkout orders**.
- **Mean Sales per Line Item:** `df['sales'].mean()` = **$229.86**
- **Average Order Value (AOV):** `df.groupby('order_id')['sales'].sum().mean()` = **$458.61**

Because orders contain an average of 2.0 items, evaluating at the transaction level is required for genuine commercial analysis.
"""))

cells.append(nbf.v4.new_code_cell("""kpis = calculate_core_kpis(df_cleaned)

kpi_table = pd.DataFrame([
    {"Business Metric": "Total Gross Revenue", "Value": f"${kpis['total_revenue']:,.2f}"},
    {"Business Metric": "Total Net Profit", "Value": f"${kpis['total_profit']:,.2f}"},
    {"Business Metric": "Overall Profit Margin", "Value": f"{kpis['overall_profit_margin_pct']}%"},
    {"Business Metric": "Total Unique Orders", "Value": f"{kpis['total_orders']:,}"},
    {"Business Metric": "Total Unique Customers", "Value": f"{kpis['total_customers']:,}"},
    {"Business Metric": "Total Units Sold", "Value": f"{kpis['total_quantity_sold']:,}"},
    {"Business Metric": "Average Order Value (AOV)", "Value": f"${kpis['avg_order_value']:,.2f}"},
    {"Business Metric": "Median Order Value", "Value": f"${kpis['median_order_value']:,.2f}"},
    {"Business Metric": "Average Profit per Order", "Value": f"${kpis['avg_profit_per_order']:,.2f}"},
    {"Business Metric": "Median Profit per Order", "Value": f"${kpis['median_profit_per_order']:,.2f}"},
    {"Business Metric": "Profitable Orders", "Value": f"{kpis['profitable_orders_count']:,} ({kpis['profitable_orders_pct']}%)"},
    {"Business Metric": "Loss-Making Orders", "Value": f"{kpis['loss_making_orders_count']:,} ({kpis['loss_making_orders_pct']}%)"}
])

display(kpi_table)
"""))

cells.append(nbf.v4.new_markdown_cell("""### Mean vs. Median & Skewness Analysis
Comparing parametric vs. non-parametric measures reveals severe right-skew in financial metrics ($S = +12.97$ for sales).
"""))

cells.append(nbf.v4.new_code_cell("""dist_stats = calculate_distribution_statistics(df_cleaned)
display(dist_stats)
"""))

# =============================================================
# 8. Exploratory Data Analysis & GroupBy
# =============================================================
cells.append(nbf.v4.new_markdown_cell("""## 8. Exploratory Data Analysis (GroupBy Foundations)

Demonstrating multi-dimensional aggregation via `groupby()`:
- `sum()`: Aggregate volume and totals
- `mean()`: Baseline transaction sizing
- `nunique()`: Entity counts (orders, customers)
"""))

# =============================================================
# 9. Category Analysis
# =============================================================
cells.append(nbf.v4.new_markdown_cell("""## 9. Category & Sub-Category Analysis
Evaluating total sales, profit, quantity, and profit margin share across categories:
"""))

cells.append(nbf.v4.new_code_cell("""cat_summary = calculate_category_metrics(df_cleaned, group_col='category')
display(cat_summary)

subcat_summary = calculate_category_metrics(df_cleaned, group_col='sub_category')
display(subcat_summary.head(10))
"""))

# =============================================================
# 10. Regional Analysis
# =============================================================
cells.append(nbf.v4.new_markdown_cell("""## 10. Regional & Segment Analysis
Evaluating geographic performance and customer segment contributions:
"""))

cells.append(nbf.v4.new_code_cell("""reg_summary = calculate_regional_metrics(df_cleaned)
display(reg_summary)

seg_summary = calculate_segment_metrics(df_cleaned)
display(seg_summary)
"""))

# =============================================================
# 11. Product Analysis & Customer Micro-Analytics
# =============================================================
cells.append(nbf.v4.new_markdown_cell("""## 11. Product & Customer Micro-Analytics
Investigating revenue leaders, profit leaders, and loss leaders:
"""))

cells.append(nbf.v4.new_code_cell("""prod_insights = calculate_product_metrics(df_cleaned, top_n=5)

print("=== TOP 5 PRODUCTS BY REVENUE ===")
display(prod_insights['top_by_sales'][['product_name', 'category', 'total_sales', 'total_profit', 'profit_margin_pct']])

print("\\n=== TOP 5 PRODUCTS BY PROFIT ===")
display(prod_insights['top_by_profit'][['product_name', 'category', 'total_sales', 'total_profit', 'profit_margin_pct']])

print("\\n=== TOP 5 MOST UNPROFITABLE PRODUCTS (LOSS-LEADERS) ===")
display(prod_insights['most_unprofitable'][['product_name', 'category', 'total_sales', 'total_profit', 'profit_margin_pct']])
"""))

cells.append(nbf.v4.new_code_cell("""cust_dist, top_customers = calculate_customer_metrics(df_cleaned, top_n=5)
display(top_customers[['customer_id', 'customer_name', 'segment', 'total_sales', 'total_profit', 'order_count', 'avg_order_value']])
"""))

# =============================================================
# 12. Time-Series Analysis
# =============================================================
cells.append(nbf.v4.new_markdown_cell("""## 12. Time-Series Analysis (Multi-Year Trajectory)
Evaluating sales growth across months and years (2014–2017) to uncover cyclical seasonality.
"""))

cells.append(nbf.v4.new_code_cell("""monthly_trend = (
    df_cleaned.groupby(['order_year', 'order_month_name', 'order_month'])
    .agg(total_sales=('sales', 'sum'), total_profit=('profit', 'sum'), order_count=('order_id', 'nunique'))
    .reset_index()
    .sort_values(by=['order_year', 'order_month'])
)
display(monthly_trend.head(12))
"""))

# =============================================================
# 13. Correlation Analysis
# =============================================================
cells.append(nbf.v4.new_markdown_cell("""## 13. Correlation Analysis

### Correlation $\\neq$ Causation
The Pearson correlation coefficient ($r$) measures linear association between two continuous variables on a scale from $-1.0$ (perfect inverse linear) to $+1.0$ (perfect direct linear). A correlation indicates linear association only and does not prove that changes in one variable cause changes in the other.
"""))

cells.append(nbf.v4.new_code_cell("""corr_matrix = calculate_correlations(df_cleaned)
display(corr_matrix)
"""))

# =============================================================
# 14. Phase 4 — Visualizations
# =============================================================
cells.append(nbf.v4.new_markdown_cell("""## 14. Phase 4 — Exploratory Data Analysis & Visualizations

Every visualization below answers a defined analytical question, with structured breakdown into **Question**, **Method**, **Observation**, and **Interpretation**.
"""))

# 14.1 Visualization objectives
cells.append(nbf.v4.new_markdown_cell("""### 14.1 Visualization Objectives
1. Profile categorical and regional revenue/profit dynamics.
2. Uncover multi-year temporal trends and seasonality.
3. Diagnose skewness and test normalizations for ML pipelines.
4. Statistically evaluate correlation relationships and discounting impact.
"""))

# 14.2 Sales by Category
cells.append(nbf.v4.new_markdown_cell("""### 14.2 Sales by Category
- **Question:** Which product category generates the most revenue?
- **Method:** Aggregating total $\\sum(\\text{sales})$ by `category`.
- **Observation:** Technology generated the largest revenue share ($836,154.03, 36.4%), followed by Furniture ($741,999.80, 32.3%) and Office Supplies ($719,047.03, 31.3%).
- **Interpretation:** Revenue is distributed across all three categories, with Technology slightly leading.
"""))

cells.append(nbf.v4.new_code_cell("""fig1 = plot_sales_by_category(df_cleaned, output_path=os.path.join(project_root, 'outputs', 'figures', '01_sales_by_category.png'))
plt.show()
"""))

# 14.3 Profit by Category
cells.append(nbf.v4.new_markdown_cell("""### 14.3 Profit by Category
- **Question:** How does profitability differ across categories?
- **Method:** Aggregating total $\\sum(\\text{profit})$ by `category`.
- **Observation:** Technology delivered **50.79% of all net profit ($145.45k)** with a 17.40% margin. Office Supplies contributed **42.77% ($122.49k)** with a 17.04% margin. In contrast, Furniture generated only **6.44% of total profit ($18.45k)** with a 2.49% margin.
- **Interpretation:** Despite contributing nearly a third of total revenue, Furniture is severely margin-dilutive due to heavy discounting in Tables and Bookcases.
"""))

cells.append(nbf.v4.new_code_cell("""fig2 = plot_profit_by_category(df_cleaned, output_path=os.path.join(project_root, 'outputs', 'figures', '02_profit_by_category.png'))
plt.show()
"""))

# 14.4 Sales by Region
cells.append(nbf.v4.new_markdown_cell("""### 14.4 Sales by Region
- **Question:** How is revenue distributed geographically?
- **Method:** Aggregating total $\\sum(\\text{sales})$ by `region`.
- **Observation:** West is the highest-revenue region ($725.46k, 31.6%), followed by East ($678.78k, 29.5%), Central ($501.24k, 21.8%), and South ($391.72k, 17.1%).
- **Interpretation:** Coastal regions (West and East) drive over 61% of total enterprise volume.
"""))

cells.append(nbf.v4.new_code_cell("""fig3 = plot_sales_by_region(df_cleaned, output_path=os.path.join(project_root, 'outputs', 'figures', '03_sales_by_region.png'))
plt.show()
"""))

# 14.5 Monthly Sales Trend
cells.append(nbf.v4.new_markdown_cell("""### 14.5 Monthly Sales Trend
- **Question:** How does revenue change over time across multi-year cycles?
- **Method:** Chronological monthly aggregation by `order_year_month` with a 3-month moving average.
- **Observation:** Clear upward long-term trend across 2014–2017 with elevated sales in September and fourth-quarter peaks in November and December.
- **Interpretation:** Retail volume accelerates in late summer and peaks during the holiday shopping quarter.
"""))

cells.append(nbf.v4.new_code_cell("""fig4 = plot_monthly_sales_trend(df_cleaned, output_path=os.path.join(project_root, 'outputs', 'figures', '04_monthly_sales_trend.png'))
plt.show()
"""))

# 14.6 Sales Distribution
cells.append(nbf.v4.new_markdown_cell("""### 14.6 Sales Distribution
- **Question:** How are individual line-item sales distributed?
- **Method:** Dual-panel histogram displaying raw linear scale vs. $\\log_{10}(\\text{sales})$ scale.
- **Observation:** Sales are strongly right-skewed ($\\text{Mean}=\\$229.86 \\gg \\text{Median}=\\$54.49, S=+12.97$). A logarithmic transformation substantially reduces the observed skewness and produces a more symmetric distribution.
- **Interpretation:** A log transformation may be evaluated in future predictive modeling.
"""))

cells.append(nbf.v4.new_code_cell("""fig5 = plot_sales_distribution(df_cleaned, output_path=os.path.join(project_root, 'outputs', 'figures', '05_sales_distribution.png'))
plt.show()
"""))

# 14.7 Top Products
cells.append(nbf.v4.new_markdown_cell("""### 14.7 Top Products by Revenue
- **Question:** Which individual products generate the highest sales?
- **Method:** Horizontal bar chart of top 10 products by $\\sum(\\text{sales})$.
- **Observation:** The *Canon imageCLASS 2200 Advanced Copier* leads with $61,599.82 in sales and $25,199.93 in profit (40.9% margin).
- **Interpretation:** High-ticket commercial hardware represents disproportionately large revenue and profit drivers.
"""))

cells.append(nbf.v4.new_code_cell("""fig6 = plot_top_products_by_revenue(df_cleaned, output_path=os.path.join(project_root, 'outputs', 'figures', '06_top_products_by_revenue.png'), top_n=10)
plt.show()
"""))

# 14.8 Quantity vs Sales
cells.append(nbf.v4.new_markdown_cell("""### 14.8 Quantity vs. Sales
- **Question:** Is there a relationship between quantity and sales?
- **Method:** Scatter plot with linear trendline and Pearson correlation ($r = +0.20$).
- **Observation:** Quantity and sales exhibit a weak positive linear association ($r = +0.20$), indicating that quantity alone does not explain much of the variation in line-item sales. Sales also vary across unit prices and product types.
- **Interpretation:** Price dispersion across product categories dominates overall transaction sales.
"""))

cells.append(nbf.v4.new_code_cell("""fig7 = plot_quantity_vs_sales(df_cleaned, output_path=os.path.join(project_root, 'outputs', 'figures', '07_quantity_vs_sales.png'))
plt.show()
"""))

# 14.9 Correlation Analysis Heatmap
cells.append(nbf.v4.new_markdown_cell("""### 14.9 Correlation Heatmap
- **Question:** What relationships exist among the numerical variables?
- **Method:** Annotated Seaborn heatmap of Pearson correlation coefficients.
- **Observation:** `sales` vs `profit` shows a moderate positive linear association ($r = +0.48$). `discount` vs `profit` shows a moderate negative linear association ($r = -0.22$).
- **Interpretation:** Observational correlation confirms inverse association between discounts and profit in this dataset.
"""))

cells.append(nbf.v4.new_code_cell("""fig8 = plot_correlation_heatmap(df_cleaned, output_path=os.path.join(project_root, 'outputs', 'figures', '08_correlation_heatmap.png'))
plt.show()
"""))

# 14.10 Discount vs Profitability
cells.append(nbf.v4.new_markdown_cell("""### 14.10 Discount vs. Profitability
- **Question:** How does profitability vary across discount levels?
- **Method:** Aggregating total net profit and margin % across discrete discount percentages.
- **Observation:**
  - Discounts $\\le 20\\%$ yield solid positive profits ($0\\% \\rightarrow +\\$320.99k, 20\\% \\rightarrow +\\$90.34k$).
  - Higher discount levels ($\ge 30\%$) are associated with lower aggregate profitability in this dataset ($-10.05\\%$ margin at $30\\%$, dropping to $-180.03\\%$ at $80\\%$ discount).
- **Interpretation:** Deep promotional discounting is associated with severe margin dilution.
"""))

cells.append(nbf.v4.new_code_cell("""fig9 = plot_discount_profitability(df_cleaned, output_path=os.path.join(project_root, 'outputs', 'figures', '09_discount_vs_profitability.png'))
plt.show()
"""))

# 14.11 Profit Margin by Category
cells.append(nbf.v4.new_markdown_cell("""### 14.11 Profit Margin by Category
- **Question:** How do category profit margins compare to revenue rankings?
- **Method:** Bar chart of $(\\sum\\text{profit} / \\sum\\text{sales}) \\times 100$ by `category`.
- **Observation:** Technology (17.40%) and Office Supplies (17.04%) lead in margin efficiency, while Furniture trails significantly at 2.49%.
- **Interpretation:** Revenue ranking does not mirror profitability; margin efficiency requires careful monitoring.
"""))

cells.append(nbf.v4.new_code_cell("""fig10 = plot_category_profit_margin(df_cleaned, output_path=os.path.join(project_root, 'outputs', 'figures', '10_profit_margin_by_category.png'))
plt.show()
"""))

# 14.12 Visualization Findings
cells.append(nbf.v4.new_markdown_cell("""### 14.12 Visualization Findings Summary
- Technology generated the highest total profit (50.8% share).
- Furniture had the lowest aggregate profit margin (2.49%).
- Coastal regions (West & East) drive >61% of total enterprise volume.
- Fourth quarter shows recurring annual demand peaks.
- Higher discount levels ($\ge 30\%$) were associated with lower aggregate profitability.
"""))

# =============================================================
# 15. Key Insights
# =============================================================
cells.append(nbf.v4.new_markdown_cell("""## 15. Key Insights (Evidence-Based)

1. **Technology Drives Profit Leadership:**
   - *Observed Fact:* Technology generated **$836,154.03 (36.40% of sales)** and **$145,454.95 (50.79% of total profit)** with a 17.40% net margin.
   - *Interpretation:* Enterprise technology sales represent high-margin commercial value and should remain a core growth focus.

2. **Furniture Suffers Severe Margin Dilution:**
   - *Observed Fact:* Furniture accounted for **$741,999.80 (32.30% of sales)** but delivered only **$18,451.27 (6.44% of profit)** with a slim 2.49% margin. Tables specifically operated at a net loss of **-$17,725.48**.
   - *Interpretation:* Heavy discounting and logistics overhead on bulky furniture items erode profitability.

3. **Discounting Associated with Cumulative Losses Beyond 20%:**
   - *Observed Fact:* Orders with $0\\% - 20\\%$ discounts yielded **+$421,773.08 in positive net profits**. Across all transactions with discounts $\\ge 30\\%$, cumulative losses reached **-$135,376.06**.
   - *Interpretation:* Higher discount levels are associated with lower aggregate profitability in this dataset.

4. **Late-Year and Q4 Volume Concentration:**
   - *Observed Fact:* Monthly sales consistently accelerate in September and reach annual peaks in Q4 (November and December).
   - *Interpretation:* Inventory, working capital, and supply chain logistics must ramp up starting in Q3 to support peak demand.

5. **Customer & Product Concentration Nuances:**
   - *Observed Fact:* The #1 highest revenue customer generated a **-$1,980.74 net loss**, and the #3 highest revenue product generated a **-$1,811.08 in net losses**.
   - *Interpretation:* Revenue volume does not equal profitability. Sales teams must align incentive structures with net profit contribution rather than top-line gross revenue.
"""))

# =============================================================
# 16. Conclusion
# =============================================================
cells.append(nbf.v4.new_markdown_cell("""## 16. Conclusion & Machine Learning Preparedness

### Summary of Accomplishments:
- Audited and verified zero data loss across 9,994 transaction records.
- Standardized data schemas, datetime casting, and feature engineering.
- Differentiated row-level metrics from order-level aggregations ($AOV$).
- Generated 10 publication-quality visualizations answering clear analytical questions.
- Formulated 5 evidence-based strategic business insights.

### Machine Learning Preparedness:
1. **Feature Transformations:** Apply Log1p transformations to `sales`, `profit`, and `unit_price` before fitting linear models.
2. **Binary Classification Problem:** Predicting whether an upcoming order will be profitable vs. loss-making (79.26% vs 20.40% class balance).
3. **Customer Lifetime Value (CLV) Modeling:** Utilizing RFM (Recency, Frequency, Monetary) metrics for predictive customer segmentation.
"""))

nb.cells = cells

with open('notebooks/01_exploratory_data_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print('Successfully generated complete 16-section production notebook!')
