# E-Commerce Sales Data Analysis & Insights

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-red.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-purple.svg)](https://seaborn.pydata.org/)
[![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon%20ARM64-lightgrey.svg)]()

An end-to-end, publication-grade **Exploratory Data Analysis (EDA)** and data engineering project built with Python 3.11. Designed as a foundational AI/Data Engineering portfolio project demonstrating rigorous data quality auditing, statistical distribution modeling, multi-dimensional business aggregations, publication-quality visualizations, and evidence-based strategic insights.

---

## 📊 Project Overview

Before building machine learning models (regression, classification, or deep learning), an AI Engineer must thoroughly understand data distributions, audit data quality, validate schema integrity, and engineer domain-relevant features.

This project analyzes the real-world **Superstore Sales Dataset** (9,994 transaction line items across 5,009 unique customer orders between 2014 and 2017) using lightweight, high-performance scientific computing libraries on Apple Silicon (macOS ARM64).

---

## 🎯 Objectives

1. **Data Quality & Integrity:** Perform a non-destructive data quality audit (missing values, exact duplicates, chronological validation, and IQR outlier detection).
2. **Feature Engineering:** Derive temporal attributes (`order_year`, `order_month`, `order_quarter`, `order_year_month`) and financial fields (`unit_price`).
3. **Descriptive Statistics:** Calculate parametric and non-parametric distribution statistics, analyze skewness, and contrast mean vs. median.
4. **Order-Level KPI Formulation:** Clearly differentiate individual line-item metrics from aggregated checkout order-level metrics (Average Order Value - $AOV$).
5. **Multi-Dimensional GroupBy:** Evaluate revenue, profit margins, and volume across categories, sub-categories, regions, customer segments, and products.
6. **Publication Visualizations:** Generate 10 high-resolution (300 DPI) charts answering targeted business questions.
7. **Evidence-Based Insights:** Derive data-driven business observations separated strictly into observed facts and strategic interpretations.

---

## 🗂️ Dataset

- **Dataset Source:** Sample Superstore Retail Transaction Dataset (Tableau public benchmark / Kaggle open retail data).
- **Raw File:** `data/raw/sales.csv` (Preserved 100% untouched).
- **Processed File:** `data/processed/cleaned_sales.csv`.
- **Dimensions:** 9,994 rows, 21 raw columns $\rightarrow$ 27 cleaned and enriched columns.
- **Granularity:** Transaction line-item records representing 5,009 unique orders and 793 unique customers across 49 US states.

---

## 🏗️ Data Pipeline

```text
Raw Superstore Dataset (data/raw/sales.csv)
        ↓
Data Loading & Schema Inspection (src/data_cleaning.py)
        ↓
Data Quality Assessment (Nulls, Duplicates, Boundary Checks)
        ↓
Data Cleaning (Snake_casing, Datetime Casting)
        ↓
Feature Engineering (Date attributes, Unit Price)
        ↓
Processed Dataset Export (data/processed/cleaned_sales.csv)
        ↓
Descriptive Statistics & KPI Formulation (src/analysis.py)
        ↓
Multi-Dimensional Exploratory Analysis & GroupBy
        ↓
Visualization & Correlation Modeling (src/visualization.py)
        ↓
Evidence-Based Strategic Insights (outputs/reports/eda_summary_report.md)
```

---

## 🛠️ Technology Stack

- **Language:** Python 3.11
- **Data Manipulation:** Pandas 2.0+, NumPy 1.24+
- **Data Visualization:** Matplotlib 3.7+, Seaborn 0.12+
- **Interactive Interface:** Jupyter Notebook, IPython Kernel
- **Hardware Optimization:** Apple Silicon (macOS ARM64, M2, 16 GB RAM)

---

## 📁 Project Structure

```text
E-Commerce Sales Data Analysis & Insights/
│
├── data/
│   ├── raw/
│   │   └── sales.csv                      # Raw untouched transactional dataset (2.28 MB)
│   └── processed/
│       └── cleaned_sales.csv              # Validated, enriched dataset (2.58 MB)
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # 16-section fully executed Jupyter Notebook
│
├── src/
│   ├── __init__.py                        # Package initializer
│   ├── data_cleaning.py                   # Data loading, validation, feature engineering
│   ├── analysis.py                        # KPI formulation, distribution, GroupBy metrics
│   └── visualization.py                   # 10 publication-quality plotting functions
│
├── outputs/
│   ├── figures/                           # 10 exported 300 DPI visualization PNGs
│   │   ├── 01_sales_by_category.png
│   │   ├── 02_profit_by_category.png
│   │   ├── 03_sales_by_region.png
│   │   ├── 04_monthly_sales_trend.png
│   │   ├── 05_sales_distribution.png
│   │   ├── 06_top_products_by_revenue.png
│   │   ├── 07_quantity_vs_sales.png
│   │   ├── 08_correlation_heatmap.png
│   │   ├── 09_discount_vs_profitability.png
│   │   └── 10_profit_margin_by_category.png
│   │
│   └── reports/
│       └── eda_summary_report.md          # Executive analytical summary report
│
├── generate_notebook.py                   # Automated reproducible notebook builder
├── requirements.txt                       # Minimal pinned dependencies
├── .gitignore                             # Ignores .venv, caches, checkpoints, OS files
└── README.md                              # Comprehensive project documentation
```

---

## 🧹 Data Cleaning

A core principle of this project is: **Data cleaning $\neq$ deleting unusual data.**

- **Row Preservation:** 100% of observations preserved (9,994 rows before $\rightarrow$ 9,994 rows after).
- **Missing Values & Duplicates:** Verified 0 null values and 0 duplicate records.
- **Negative Profit Retention:** 2,174 line items with negative profit were retained because they represent legitimate commercial dynamics (loss-leaders, steep promotional discounts).
- **Outlier Retention:** 1,167 high-value sales outliers were retained as genuine enterprise purchases.
- **Header Standardization:** Converted to lowercase `snake_case` (e.g., `Order ID` $\rightarrow$ `order_id`).
- **Date Conversion:** Parsed string dates into `datetime64[ns]` with zero `NaT` generation.

---

## 📈 Exploratory Data Analysis

### Mean vs. Median & Distribution Skewness
```text
Feature       Mean     Median   Std Dev    Variance     Min        Q1 (25%)  Q3 (75%)   IQR       Max        Skewness  Shape
sales         $229.86  $54.49   $623.25    388,434.46   $0.44      $17.28    $209.94    $192.66   $22,638.48  +12.97   Severe Right-Skew
profit        $28.66   $8.67    $234.26    54,877.80   -$6,599.98  $1.73     $29.36     $27.64    $8,399.98   +7.56    Severe Right-Skew / Fat Tails
quantity      3.79     3.00     2.23       4.95         1.00       2.00      5.00       3.00      14.00       +1.28    Moderate Right-Skew
unit_price    $60.92   $16.27   $142.93    20,428.26    $0.34      $5.47     $63.94     $58.47    $3,773.08   +10.78   Severe Right-Skew
discount      0.16     0.20     0.21       0.04         0.00       0.00      0.20       0.20      0.80        +1.68    Multi-Modal Spikes
```

- Sales are strongly right-skewed ($\text{Mean } \$229.86 \gg \text{Median } \$54.49, S = +12.97$). A logarithmic transformation substantially reduces the observed skewness and produces a more symmetric distribution.

---

## 📊 Visualizations

All 10 project charts are exported to `outputs/figures/` at 300 DPI:

| Chart | Focus Question | Aggregation Method |
| :--- | :--- | :--- |
| **01. Sales by Category** | Which product category generates the most revenue? | $\sum(\text{sales})$ by `category` |
| **02. Profit by Category** | How does net profitability differ across categories? | $\sum(\text{profit})$ by `category` |
| **03. Sales by Region** | How is gross revenue distributed geographically? | $\sum(\text{sales})$ by `region` |
| **04. Monthly Sales Trend** | How does revenue change over time across multi-year cycles? | $\sum(\text{sales})$ by chronological `order_year_month` |
| **05. Sales Distribution** | How are individual line-item sales distributed? | Histogram: Raw Linear vs. $\log_{10}(\text{Sales})$ |
| **06. Top 10 Products by Revenue** | Which individual products generate the highest sales? | Top 10 products by $\sum(\text{sales})$ |
| **07. Quantity vs. Sales** | Is there a relationship between quantity and sales? | Scatter plot + Trendline ($r = +0.20$) |
| **08. Correlation Heatmap** | What linear relationships exist among numerical variables? | Pearson correlation matrix ($5 \times 5$) heatmap |
| **09. Discount vs. Profitability** | How does profitability vary across discount levels? | Grouped by `discount` $\rightarrow \sum(\text{profit})$ & Margin % |
| **10. Profit Margin by Category** | How do category profit margins compare to revenue rankings? | $(\sum\text{profit} / \sum\text{sales}) \times 100$ by `category` |

---

## 💰 Business KPIs

| Business Metric | Calculated Value | Analytical Scope |
| :--- | ---: | :--- |
| **Total Gross Revenue** | **$2,297,200.86** | Sum of all sales records |
| **Total Net Profit** | **$286,397.02** | Sum of all profit records |
| **Overall Profit Margin** | **12.47%** | $(\text{Total Profit} / \text{Total Revenue}) \times 100$ |
| **Total Units Sold** | **37,873 units** | Sum of all quantities |
| **Total Unique Orders** | **5,009 orders** | Unique `order_id` checkout transactions |
| **Total Unique Customers** | **793 customers** | Unique `customer_id` accounts |
| **Average Order Value (AOV)** | **$458.61** | Mean of aggregated order-level sales |
| **Median Order Value** | **$151.96** | Median of aggregated order-level sales |
| **Average Profit per Order** | **$57.18** | Mean of aggregated order-level profit |
| **Median Profit per Order** | **$16.28** | Median of aggregated order-level profit |
| **Average Sales per Line Item** | **$229.86** | Mean of row-level sales (`df['sales'].mean()`) |
| **Profitable Orders** | **3,970 (79.26%)** | Orders with aggregated net profit $> 0$ |
| **Loss-Making Orders** | **1,022 (20.40%)** | Orders with aggregated net profit $< 0$ |
| **Break-Even Orders** | **17 (0.34%)** | Orders with aggregated net profit $= 0$ |

---

## 🔎 Key Insights

1. **Technology Drives Profit Leadership:**
   - *Observed Fact:* Technology generated **$836,154.03 (36.40% of sales)** and **$145,454.95 (50.79% of total net profit)** with a 17.40% margin.
   - *Interpretation:* Commercial technology hardware represents high-margin business value.
2. **Furniture Suffers Substantial Margin Dilution:**
   - *Observed Fact:* Furniture accounted for **$741,999.80 (32.30% of sales)** but delivered only **$18,451.27 (6.44% of profit)** with a slim 2.49% margin. Tables operated at an aggregate net loss of **-$17,725.48**.
   - *Interpretation:* Heavy discounting and logistics overhead on bulky furniture items severely erode net profitability.
3. **Discount Association with Heavy Losses Beyond 20%:**
   - *Observed Fact:* Orders with $0\% - 20\%$ discounts produced **+$421,773.08 in positive net profit**, whereas orders with discounts $\ge 30\%$ accumulated **-$135,376.06 in net losses**.
   - *Interpretation:* Higher discount levels are strongly associated with commercial margin erosion in this dataset.
4. **Late-Year and Q4 Volume Concentration:**
   - *Observed Fact:* Monthly sales consistently accelerate in September and reach their annual peak during the fourth quarter (November and December).
   - *Interpretation:* Operational demand planning, working capital, and supply chain logistics must ramp up prior to Q4.
5. **Revenue Volume Does Not Guarantee Profitability:**
   - *Observed Fact:* The #1 highest revenue customer generated a **-$1,980.74 net loss**, and the #3 highest revenue product generated a **-$1,811.08 net loss**.
   - *Interpretation:* Sales evaluation and incentive structures must track net profit contribution alongside top-line gross revenue.

---

## 📋 Analytical Report

A comprehensive executive summary report is available at:
👉 **[`outputs/reports/eda_summary_report.md`](outputs/reports/eda_summary_report.md)**

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd "E-Commerce Sales Data Analysis & Insights"
```

### 2. Create and activate a Python 3.11 virtual environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install lightweight dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Option A: Run the End-to-End Pipeline via Python Modules
```bash
# 1. Run Data Cleaning & Quality Pipeline
python -c "from src.data_cleaning import clean_dataset; clean_dataset()"

# 2. Run Descriptive Statistics & KPIs
python -c "import pandas as pd; from src.analysis import calculate_core_kpis; df = pd.read_csv('data/processed/cleaned_sales.csv'); print(calculate_core_kpis(df))"

# 3. Generate all 10 Publication Visualizations
python -c "import pandas as pd; from src.visualization import generate_all_visualizations; df = pd.read_csv('data/processed/cleaned_sales.csv'); generate_all_visualizations(df)"
```

### Option B: Launch Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

---

## 🧪 Validation

Before publishing, the repository passed an automated end-to-end verification:
- `0` execution errors
- `0` broken imports
- `0` missing output files
- Verified exact schema consistency between raw and processed data.

---

## ⚠️ Limitations

- **Observational Data:** Statistical findings reflect observed historical relationships and do not prove causality.
- **Confounding Factors:** Product category, manufacturer costs, and shipping logistics may confound discount and margin relationships.
- **Linearity:** Pearson correlation ($r$) measures only linear relationships.
- **No Machine Learning Models:** This project is strictly an exploratory data analysis foundation.

---

## 🔮 Future Improvements

Planned future work for subsequent AI Engineering project stages:
1. **Machine Learning Regression:** Build revenue and pricing prediction models.
2. **Binary Classification:** Predict order profitability (profitable vs. loss-making orders).
3. **Time-Series Forecasting:** Implement ARIMA/Prophet models for Q4 seasonal demand forecasting.
4. **Customer Segmentation:** Apply unsupervised K-Means/RFM clustering for customer cohort analysis.
5. **Interactive Dashboard:** Build a lightweight Streamlit/Dash interactive interface.

---

## 👨‍💻 Skills Demonstrated

- **Python Programming:** Clean modular architecture, type hinting, PEP 8 compliance, error handling.
- **Pandas:** Tabular manipulation, datetime parsing, feature derivation, multi-dimensional `groupby()`.
- **NumPy:** Vectorized math, non-parametric percentiles, polynomial trendline estimation.
- **Statistics:** Parametric & non-parametric profiling, IQR outlier detection, skewness diagnosis, Pearson correlation.
- **Data Visualization:** Matplotlib & Seaborn, publication-quality styling, dual-scale distributions, heatmaps.
- **Data Engineering Practices:** Data quality auditing, non-destructive data cleaning, reproducible code structure.
