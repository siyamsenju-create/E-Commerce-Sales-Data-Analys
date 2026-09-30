# E-Commerce Sales Data Analysis & Insights

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-red.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-purple.svg)](https://seaborn.pydata.org/)
[![Pytest](https://img.shields.io/badge/Pytest-9.1+-brightgreen.svg)](https://docs.pytest.org/)
[![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon%20ARM64-lightgrey.svg)]()

An end-to-end, production-quality **Exploratory Data Analysis (EDA)**, data quality engineering, and statistical analytics portfolio project built in Python 3.11. This project demonstrates rigorous schema validation, parametric and non-parametric distribution profiling, multi-dimensional business aggregations, publication-quality visualizations, and machine learning readiness.

---

## 📊 Overview

Before architecting machine learning models (regression, classification, clustering, or forecasting), an AI/Data Engineer must understand underlying data distributions, audit data quality, validate schema integrity, and extract domain-relevant features without introducing target leakage or destructive record deletion.

This project analyzes the real-world **Superstore Sales Dataset** ($9,994$ transaction line items across $5,009$ customer checkout orders from $2014$ to $2017$) using a clean, modular Python codebase and a lightweight scientific computing stack optimized for macOS Apple Silicon.

---

## 🎯 Objectives

1. **Data Quality & Integrity:** Conduct a non-destructive audit (missing values, exact duplicates, chronological validation, and IQR outlier detection).
2. **Feature Engineering:** Derive temporal attributes (`order_year`, `order_month`, `order_quarter`, `order_year_month`), fulfillment latency (`shipping_days`), and item pricing (`unit_price`).
3. **Statistical Distribution Profiling:** Calculate parametric and non-parametric summary statistics, quantify skewness, and contrast mean vs. median.
4. **Order-Level vs. Line-Item Precision:** Differentiate row-level sales metrics from aggregated checkout-level metrics (Average Order Value - $AOV$).
5. **Multi-Dimensional GroupBy:** Evaluate revenue, profit margins, and volume across product categories, geographic regions, customer segments, and products.
6. **Publication Visualizations:** Generate 10 high-resolution (300 DPI) charts answering targeted analytical questions.
7. **Evidence-Based Insights:** Derive data-driven business findings strictly separated into observed facts and strategic interpretations.
8. **Machine Learning Preparedness:** Outline downstream predictive problems and establish data leakage prevention standards.

---

## ❓ Business Questions

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

---

## 🗂️ Dataset

- **Dataset Source:** Sample Superstore Retail Transaction Dataset (Tableau public benchmark / Kaggle open retail data).
- **Raw File:** `data/raw/sales.csv` (Preserved 100% untouched, 2.28 MB).
- **Processed File:** `data/processed/cleaned_sales.csv` (2.58 MB).
- **Dimensions:** 9,994 rows, 21 raw columns $\rightarrow$ 28 cleaned and enriched columns.
- **Granularity:** Transaction line items representing 5,009 unique checkout orders from 793 unique customers across 49 US states.
- **Licensing / Attribution:** Standard educational retail benchmark dataset. Direct licensing should be verified before external commercial redistribution.

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Data Manipulation:** Pandas 2.0+, NumPy 1.24+
- **Data Visualization:** Matplotlib 3.7+, Seaborn 0.12+
- **Testing & Quality Assurance:** Pytest 9.0+
- **Interactive Computing:** Jupyter Notebook, IPython Kernel
- **Environment:** CPU-only lightweight architecture on macOS Apple Silicon (ARM64)

---

## 🏛️ Project Architecture

```text
Raw Tabular Data (data/raw/sales.csv)
        ↓
Data Loading & Schema Validation (src/data_cleaning.py)
        ↓
Non-Destructive Quality Audit (Missing Values, Duplicates, Boundary Constraints)
        ↓
Data Cleaning & Type Standardization (snake_case, datetime64[ns])
        ↓
Feature Engineering (Date Components, Fulfillment Duration, Unit Price)
        ↓
Processed Dataset Export (data/processed/cleaned_sales.csv)
        ↓
Enterprise KPI Formulation & Statistical Profiling (src/analysis.py)
        ↓
Multi-Dimensional GroupBy Aggregations & Correlation Modeling
        ↓
Publication Figure Generation at 300 DPI (src/visualization.py)
        ↓
Automated Test Verification (tests/) & Executive Reporting (outputs/reports/)
```

---

## 📁 Project Structure

```text
E-Commerce Sales Data Analysis & Insights/
│
├── data/
│   ├── raw/
│   │   └── sales.csv                      # Untouched transactional dataset (9,994 rows, 2.28 MB)
│   └── processed/
│       └── cleaned_sales.csv              # Validated, enriched dataset (28 columns, 2.58 MB)
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # 21-section executed Jupyter Notebook
│
├── src/
│   ├── __init__.py                        # Package initializer
│   ├── data_cleaning.py                   # Loading, schema validation, feature engineering
│   ├── analysis.py                        # KPI formulation, distribution, GroupBy metrics
│   └── visualization.py                   # 10 publication-quality 300 DPI plotting functions
│
├── tests/
│   ├── __init__.py                        # Test package initializer
│   ├── test_data_cleaning.py              # Schema, date casting, and validation tests
│   ├── test_analysis.py                   # KPI formulas and group-by verification tests
│   └── test_edge_cases.py                 # Empty DF, invalid inputs, and anomaly tests
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
│       └── eda_summary_report.md          # Comprehensive executive summary report
│
├── generate_notebook.py                   # Automated notebook generator script
├── main.py                                # End-to-end pipeline runner
├── requirements.txt                       # Minimal pinned reproducible dependencies
├── .gitignore                             # Ignores .venv, caches, checkpoints, OS files
└── README.md                              # Comprehensive project documentation
```

---

## 🔄 Data Pipeline

The automated pipeline can be executed end-to-end via:
```bash
python main.py
```
This orchestrates the 3 core phases:
1. `clean_dataset()`: Loads raw data, validates constraints, engineers features, and saves `data/processed/cleaned_sales.csv`.
2. `calculate_core_kpis()`: Computes order-level and line-item enterprise metrics.
3. `generate_all_visualizations()`: Produces all 10 high-resolution 300 DPI figures in `outputs/figures/`.

---

## 🧹 Data Cleaning

A primary tenet of this data engineering workflow is: **Data cleaning $\neq$ deleting unusual data.**

- **Row Preservation:** 100.0% of records preserved (9,994 raw rows $\rightarrow$ 9,994 processed rows).
- **Missing Values:** Verified 0 null values across all features.
- **Exact Duplicates:** Verified 0 duplicate records.
- **Negative Profit Retention:** 2,174 line items with negative profit were retained as legitimate commercial dynamics resulting from steep promotional discounting.
- **Outlier Retention:** 1,167 high-value sales outliers were retained as genuine commercial transactions.
- **Date Validation:** Enforced `ship_date >= order_date` across all rows with 0 chronological anomalies.

---

## ⚙️ Feature Engineering

| Feature | Data Type | Formula / Origin | Analytical Purpose |
| :--- | :--- | :--- | :--- |
| `order_year` | Integer | `order_date.dt.year` | Multi-year growth and annual comparisons |
| `order_month` | Integer | `order_date.dt.month` | Monthly seasonality |
| `order_month_name` | String | `order_date.dt.strftime('%b')` | Visual chart labeling |
| `order_quarter` | Integer | `order_date.dt.quarter` | Quarterly tracking |
| `order_year_month` | String | `order_date.dt.to_period('M')` | Multi-year time-series plotting |
| `shipping_days` | Integer | `(ship_date - order_date).dt.days` | Fulfillment latency profiling |
| `unit_price` | Float | $\text{round}(\text{sales} / \text{quantity}, 2)$ | Baseline item pricing independent of quantity |

---

## 💰 Key KPIs

| Business Metric | Calculated Value | Scope / Calculation Method |
| :--- | ---: | :--- |
| **Total Gross Revenue** | **$2,297,200.86** | Sum of all sales records |
| **Total Net Profit** | **$286,397.02** | Sum of all profit records |
| **Overall Profit Margin** | **12.47%** | $(\text{Total Profit} / \text{Total Revenue}) \times 100$ |
| **Total Units Sold** | **37,873 units** | Sum of all quantities |
| **Total Unique Orders** | **5,009 orders** | Unique `order_id` checkout transactions |
| **Total Unique Customers** | **793 customers** | Unique `customer_id` accounts |
| **Average Order Value (AOV)** | **$458.61** | Mean of aggregated order-level sales ($\text{order\_sales}$) |
| **Median Order Value** | **$151.96** | Median of aggregated order-level sales |
| **Average Profit per Order** | **$57.18** | Mean of aggregated order-level profit |
| **Median Profit per Order** | **$16.28** | Median of aggregated order-level profit |
| **Mean Sales per Line Item** | **$229.86** | Mean of individual row-level sales (`df['sales'].mean()`) |
| **Median Sales per Line Item** | **$54.49** | Median of row-level sales |
| **Profitable Orders** | **3,970 (79.26%)** | Orders with aggregated net profit $> 0$ |
| **Loss-Making Orders** | **1,022 (20.40%)** | Orders with aggregated net profit $< 0$ |
| **Break-Even Orders** | **17 (0.34%)** | Orders with aggregated net profit $= 0$ |

> **Educational Distinction:**  
> AOV ($458.61) is calculated at the **order level** (grouping by `order_id` and summing sales per basket). The mean sales per line item is **$229.86**. Because orders contain an average of 2.0 items, evaluating at the basket level is essential for commercial analytics.

---

## 🔎 Key Findings

1. **Technology Generates Majority Profitability:**
   - *Observed Fact:* Technology generated **$836,154.03 (36.40% of sales)** and **$145,454.95 (50.79% of total net profits)** with a **17.40% margin**.
   - *Interpretation:* Commercial technology equipment represents high-margin enterprise value.
2. **Furniture Suffers Margin Dilution:**
   - *Observed Fact:* Furniture accounted for **$741,999.80 (32.30% of sales)** but yielded only **$18,451.27 (6.44% of profit)** with a **2.49% margin**. Tables operated at an aggregate net loss of **-$17,725.48**.
   - *Interpretation:* Heavy discounting in specific furniture sub-categories coincided with severe margin erosion.
3. **Discount Levels Associated with Commercial Losses Beyond 20%:**
   - *Observed Fact:* Discounts $\le 20\%$ accumulated **+$421,773.08 in profit**, whereas discounts $\ge 30\%$ accumulated **-$135,376.06 in cumulative net losses**.
   - *Interpretation:* Higher discount levels were associated with lower aggregate profitability in this dataset.
4. **Late-Year & Q4 Demand Acceleration:**
   - *Observed Fact:* Monthly sales accelerate in September and reach annual peaks in the fourth quarter (November and December).
   - *Interpretation:* Operational demand planning and inventory allocation must ramp up ahead of Q4.
5. **Revenue Volume Does Not Guarantee Net Margin:**
   - *Observed Fact:* The #1 highest-revenue customer (Sean Miller: $25,043.05 sales across 5 orders) produced a **-$1,980.74 net loss**, and the #3 revenue product (Cisco TelePresence System: $22,638.48 sales) generated a **-$1,811.08 net loss**.
   - *Interpretation:* Commercial incentive structures must track net profit contribution alongside top-line gross revenue.

---

## 📊 Visualizations

All 10 project charts are exported to `outputs/figures/` at 300 DPI:

| Figure | Chart Name | Focus Question |
| :---: | :--- | :--- |
| **01** | `01_sales_by_category.png` | Which product category generates the most revenue? |
| **02** | `02_profit_by_category.png` | How does net profitability differ across categories? |
| **03** | `03_sales_by_region.png` | How is gross revenue distributed geographically? |
| **04** | `04_monthly_sales_trend.png` | How does revenue change over time across multi-year cycles? |
| **05** | `05_sales_distribution.png` | How are individual line-item sales distributed? (Linear vs. $\log_{10}$) |
| **06** | `06_top_products_by_revenue.png` | Which individual products generate the highest sales? |
| **07** | `07_quantity_vs_sales.png` | Is there a relationship between quantity and sales? ($r = +0.2008$) |
| **08** | `08_correlation_heatmap.png` | What linear relationships exist among numerical variables? |
| **09** | `09_discount_vs_profitability.png` | How does profitability vary across discount levels? |
| **10** | `10_profit_margin_by_category.png` | How do category profit margins compare to revenue rankings? |

---

## 📐 Statistical Analysis

```text
Feature        Count    Mean     Median   Std Dev    Variance     Min        Q1 (25%)  Q3 (75%)   IQR       Max        Skewness  Shape
sales          9,994    $229.86  $54.49   $623.25    388,434.46   $0.44      $17.28    $209.94    $192.66   $22,638.48  +12.97   Strongly Right-Skewed
profit         9,994    $28.66   $8.67    $234.26    54,877.80   -$6,599.98  $1.73     $29.36     $27.64    $8,399.98   +7.56    Strongly Right-Skewed / Fat Tails
quantity       9,994    3.79     3.00     2.23       4.95         1.00       2.00      5.00       3.00      14.00       +1.28    Moderately Right-Skewed
unit_price     9,994    $60.92   $16.27   $142.93    20,428.26    $0.34      $5.47     $63.94     $58.47    $3,773.08   +10.78   Strongly Right-Skewed
discount       9,994    0.16     0.20     0.21       0.04         0.00       0.00      0.20       0.20      0.80        +1.68    Multi-Modal Spikes
shipping_days  9,994    3.96     4.00     1.75       3.05         0.00       3.00      5.00       2.00      7.00        -0.20    Fairly Symmetric
```

- **Skewness Characterization:** The raw sales distribution is strongly right-skewed ($\text{Mean } \$229.86 \gg \text{Median } \$54.49, S = +12.97$). A logarithmic transformation produces a more symmetric representation.

---

## 🔗 Correlation Analysis

### Pearson Correlation Coefficients ($r$):
```text
               sales  quantity  discount    profit  unit_price
sales       1.000000  0.200795 -0.028190  0.479064    0.889376
quantity    0.200795  1.000000  0.008623  0.066253   -0.003148
discount   -0.028190  0.008623  1.000000 -0.219487   -0.032804
profit      0.479064  0.066253 -0.219487  1.000000    0.468311
unit_price  0.889376 -0.003148 -0.032804  0.468311    1.000000
```

> **Correlation $\neq$ Causation:**  
> Pearson correlation measures linear association between continuous variables. In this observational dataset, discount and profit exhibit a moderate negative linear association ($r = -0.2195$), while sales and unit price exhibit a strong positive correlation ($r = +0.8894$).

---

## 🤖 Machine Learning Readiness

### Potential Predictive Problem Formulations:
1. **Binary Classification (Order Profitability):**
   - Predict whether a checkout order will yield net positive profit ($\text{profitable\_order} \in \{0, 1\}$).
   - Target distribution: 79.26% profitable vs. 20.40% loss-making.
2. **Regression (Order-Level Profit):**
   - Predict continuous net profit per order.
3. **Customer Analytics (RFM Segmentation):**
   - Recency, Frequency, Monetary clustering via unsupervised K-Means.

### Data Leakage Awareness & Prevention:
- **Leakage Safeguards:** Features derived after checkout completion (e.g., actual fulfillment shipping days, post-order discount adjustments, line-item profit) must not be included when predicting order outcomes at basket creation time.
- **Preprocessing Protocol:** All transformers (log scaling, one-hot encoders, standard scalers) must be fit strictly on training splits.

---

## 💻 Installation

### 1. Clone the repository
```bash
git clone https://github.com/siyamsenju-create/E-Commerce-Sales-Data-Analys.git
cd "E-Commerce-Sales-Data-Analys"
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

## 🚀 Running the Project

### Option A: Run the End-to-End Pipeline
```bash
python main.py
```

### Option B: Launch the Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

---

## 🧪 Running Tests

Execute the automated test suite with pytest:
```bash
pytest -v
```
All 19 tests validate data loading, schema integrity, datetime casting, derived feature calculations, business KPI exactness, category aggregations, and edge case resilience.

---

## 🔁 Reproducibility

- **Clean-State Execution:** Tested from a fresh virtual environment.
- **Hardware Agnostic:** Optimized for lightweight execution on Apple Silicon (M2, 16 GB RAM) and CPU-only architectures without external GPU, Docker, or database dependencies.
- **Deterministic:** Non-destructive data cleaning guarantees repeatable analytical outputs.

---

## ⚠️ Limitations

- **Observational Study:** Findings reflect historical empirical associations and do not prove causality.
- **Omitted Cost Variables:** The dataset does not contain detailed freight rates, warehouse handling costs, or supplier acquisition costs.
- **Linear Correlation Scope:** Pearson correlation captures linear relationships only.

---

## 🔮 Future Improvements

1. **Statistical Hypothesis Testing:** Conduct A/B permutation tests on promotional discount tiers.
2. **Predictive Modeling Pipeline:** Train baseline classification models for order profitability with zero target leakage.
3. **Interactive Dashboard:** Build a lightweight Streamlit interface for interactive exploratory filtering.

---

## 👤 Author

**AI / Data Engineering Portfolio Project**  
*Rigorous Data Quality Auditing, Statistical Modeling, and Business KPI Formulation*
