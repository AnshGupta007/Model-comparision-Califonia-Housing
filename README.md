# 🏠 Feature Engineering, Model Optimization & Performance Comparison

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red?logo=jupyter)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An end-to-end **Machine Learning pipeline** built for an AI/ML internship task. Trains and compares three regression models on the **California Housing dataset** to predict median house prices.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Pipeline Steps](#-pipeline-steps)
- [Models Evaluated](#-models-evaluated)
- [Results](#-results)
- [Deliverables](#-deliverables)
- [Getting Started](#-getting-started)
- [Key Learnings](#-key-learnings)
- [License](#-license)

---

## 🎯 Overview

This project demonstrates a complete ML workflow:

```
Data Loading → Preprocessing → Scaling → Train/Test Split → Model Training → Evaluation → Visualization → Persistence
```

Three models are trained and compared using RMSE and R² metrics. The best model is persisted for reuse.

---

## 🛠 Tech Stack

| Library | Purpose |
|---------|---------|
| **pandas** / **NumPy** | Data manipulation & numerical computing |
| **scikit-learn** | Dataset, preprocessing, models, evaluation metrics |
| **matplotlib** / **seaborn** | Static & statistical visualizations |
| **joblib** | Model serialization / persistence |

---

## 📊 Dataset

**Source:** [`sklearn.datasets.fetch_california_housing`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

| Property | Value |
|----------|-------|
| Samples | 20,640 |
| Features | 8 |
| Target | `HousePrice` (median house value in $100,000s) |
| Origin | 1990 U.S. Census (block groups) |

**Features:** `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`

---

## 🔄 Pipeline Steps

| Step | Description |
|------|-------------|
| **1. Setup** | Import libraries, configure styling |
| **2. Data Loading** | Load dataset, rename target, inspect with `head()` / `info()` / `describe()` |
| **3. Scaling** | `StandardScaler` → zero mean, unit variance |
| **4. Split** | 80% training / 20% testing (`random_state=42`) |
| **5. Training** | Fit 3 models on scaled training data |
| **6. Evaluation** | RMSE & R² comparison in a sorted DataFrame |
| **7. Visualization** | 3-panel figure: scatter, bar chart, residual histogram |
| **8. Persistence** | `joblib.dump()` best model |
| **9. Insights** | Correlation heatmap, feature importance, prediction demo |

---

## 🤖 Models Evaluated

| Model | Configuration | Rationale |
|-------|--------------|-----------|
| **Linear Regression** | `LinearRegression()` | OLS baseline |
| **Ridge Regression** | `Ridge(alpha=1.0)` | L2 regularization to reduce overfitting |
| **Decision Tree** | `DecisionTreeRegressor(max_depth=5)` | Captures non-linear relationships |

---

## 📈 Results

| Model | RMSE | R-squared | Rank |
|-------|------|-----------|:----:|
| **Decision Tree** ✅ | **0.7242** | **0.5997** | 🥇 |
| Ridge Regression | 0.7456 | 0.5758 | 🥈 |
| Linear Regression | 0.7456 | 0.5758 | 🥉 |

**Best Model:** Decision Tree Regressor (`max_depth=5`)

- **Lowest RMSE** (0.7242) → Most accurate predictions
- **Highest R²** (0.5997) → Explains ~60% of price variance
- Outperforms linear models by capturing non-linear feature interactions

### Visualization Preview

The notebook generates a 3-panel figure:
- **Panel 1:** Actual vs Predicted scatter (with perfect-prediction diagonal)
- **Panel 2:** Side-by-side RMSE / R² bar chart (twin y-axes)
- **Panel 3:** Residual distribution histogram + KDE curve

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

### Run the Notebook

```bash
jupyter notebook AI_ML_Task2_Model_Comparison.ipynb
```

Then: **Kernel → Restart & Clear Output → Run All**

### Run the Standalone Script

```bash
python run_pipeline.py
```

Output: `best_housing_model.pkl` + `performance_plots.png`

### Export Report to PDF

```bash
# Using pandoc
pandoc AI_ML_Task2_Report.md -o AI_ML_Task2_Report.pdf --pdf-engine=pdflatex
```

---

## 💡 Key Learnings

- **Feature scaling** is essential for linear models (coefficient sensitivity) but irrelevant for tree-based models
- **Decision Trees** naturally capture feature interactions that linear models miss
- **MedInc** (median income) is the dominant predictor — 77% feature importance in the Decision Tree
- **Regularization** (Ridge) had minimal impact at `alpha=1.0` on this dataset
- **Ensemble methods** (Random Forest, XGBoost) and **hyperparameter tuning** would likely improve performance further

---

