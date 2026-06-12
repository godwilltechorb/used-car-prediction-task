# used Car Price Prediction
## End-to-End Linear Regression Project

---

## Project Overview

A complete machine learning regression project simulating the role of a junior data scientist
at an online car marketplace. The goal: **automatically predict the selling price of a used car**
based on its specifications.

| Property | Detail |
|----------|--------|
| Dataset | `Car_data.csv` — 301 rows, 9 columns (CarDekho dataset) |
| Target | `Selling_Price` (in Lakhs ₹) |
| Algorithm | Linear Regression |
| Evaluation | MAE, RMSE, R² |

---

## Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| Car_Name | Categorical | Car model/make |
| Year | Numerical | Year manufactured |
| **Selling_Price** | **Float** | **TARGET — price sold at (Lakhs ₹)** |
| Present_Price | Float | Current showroom price (Lakhs ₹) |
| Kms_Driven | Integer | Kilometres driven |
| Fuel_Type | Categorical | Petrol / Diesel / CNG |
| Seller_Type | Categorical | Dealer / Individual |
| Transmission | Categorical | Manual / Automatic |
| Owner | Integer | Number of previous owners |

---

## Steps Taken

### 1. Problem Understanding
- Defined business context: help an online marketplace estimate fair used-car prices
- Identified `Selling_Price` as the regression target
- Listed all 8 input features and their roles

### 2. Data Exploration (EDA)
- Inspected shape, dtypes, and first rows (`df.info()`, `df.describe()`, `df.head()`)
- Confirmed **zero missing values** across all 301 rows
- Detected **2 duplicate rows** (removed in preprocessing)
- Analysed value counts for all categorical columns
- Visualised `Selling_Price` distribution → right-skewed (most cars under ₹10L)
- Scatter plots: `Present_Price` vs price, `Year` vs price, `Kms_Driven` vs price
- Boxplots: price by `Fuel_Type`, `Seller_Type`, `Transmission`
- IQR outlier detection: 17 price outliers identified (retained — real data, not errors)

### 3. Data Preprocessing (Complete Pipeline)
1. **Checked dataset structure** — `df.info()`
2. **Removed 2 duplicate rows** — `df.drop_duplicates()`
3. **Verified data types** — all correct, no coercion needed
4. **Standardised text** — strip + title case on all categorical columns
5. **Handled missing values** — defensive median/mode imputation (none existed)
6. **Engineered `Car_Age`** = 2024 − Year (more interpretable than raw Year)
7. **Label-encoded** all categoricals: `Car_Name`, `Fuel_Type`, `Seller_Type`, `Transmission`
8. **Final verification** — `df.info()` + `df.isnull().sum()`
9. Correlation heatmap drawn post-encoding

### 4. Model Building
- Defined X (8 features) and y (`Selling_Price`)
- 80/20 train-test split (`random_state=42`)
- `StandardScaler` applied (fit on train only — no data leakage)
- `LinearRegression` trained on scaled training data

### 5. Model Evaluation
- Predictions generated on the unseen test set
- Metrics: MAE, RMSE, R²
- Plots: Actual vs Predicted scatter, Residuals plot, Residuals histogram

### 6. Prediction
- Selling price predicted for 5 test-set cars
- Results displayed with actual, predicted, error, and accuracy %

### 7. Insights
- Feature importance via coefficient analysis
- Written answers to: "What factors most influence price?" and "What surprised you?"

---

## Results

| Metric | Score | Interpretation |
|--------|-------|----------------|
| MAE | ₹1.55 Lakhs | Average prediction error |
| RMSE | ₹2.59 Lakhs | Penalised error (includes large misses) |
| **R² Score** | **0.7398 (73.98%)** | Model explains ~74% of price variation |

> **Grade: Good** (R² 0.70–0.79 = Good; 0.80–0.89 = Very Good; 0.90+ = Excellent)

---

## Key Findings

| Rank | Feature | Direction | Coefficient |
|------|---------|-----------|-------------|
| 1 | Present_Price | ↑ raises price | +3.89 |
| 2 | Car_Age | ↓ lowers price | −0.99 |
| 3 | Seller_Type | ↓ (individual lowers) | −0.62 |
| 4 | Transmission | ↓ (manual lowers) | −0.53 |
| 5 | Fuel_Type | ↓ (petrol/CNG lowers) | −0.49 |

---

## How to Run

### Google Colab (Recommended)
1. Open [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload Notebook** → upload `car_price_prediction.ipynb`
3. Upload `car_data_jtrjuw.csv` via the **Files sidebar** (folder icon on left)
4. **Runtime → Run All**
5. `car_price_cleaned.csv` will be saved automatically after the final cell

### Local (Jupyter)
```bash
git clone https://github.com/godwill udoh/car-price-prediction.git
cd car-price-prediction
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook car_price_prediction.ipynb
```

---

## Repository Contents

```
car-price-prediction/
├── car_price_prediction.ipynb   ← Main notebook (47 cells, all 7 sections)
├── Car_data_jrtjuw.csv                 ← Original dataset from LMS
├── car_price_cleaned.csv        ← Cleaned + encoded dataset (auto-generated)
└── README.md                    ← This file
```

---

## Author

**GODWILL UDOH**
ihifix scholar
GitHub: `https://github.com/godwill udoh/car-price-prediction`
