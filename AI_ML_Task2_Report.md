# AI/ML Internship Task 2 — Model Comparison Report

## Section 1: Methodology

### Dataset Description & Source
The California Housing dataset is sourced from `sklearn.datasets.fetch_california_housing()` and originates from the 1990 U.S. census. It contains **20,640 samples** and **8 features** representing block groups, the smallest geographical unit for which census data is published. The target variable is the median house value (`MedHouseVal`), renamed to `HousePrice` for clarity, expressed in hundreds of thousands of USD.

**Features:**
- `MedInc` — Median income (in tens of thousands of USD)
- `HouseAge` — Median house age
- `AveRooms` — Average rooms per household
- `AveBedrms` — Average bedrooms per household
- `Population` — Block group population
- `AveOccup` — Average household occupancy
- `Latitude` / `Longitude` — Geographic coordinates

### Preprocessing Steps
- **StandardScaler** was applied to all features to ensure zero mean and unit variance. This is critical for Linear Regression and Ridge Regression, which are sensitive to feature magnitudes. Without scaling, features with larger ranges (e.g., Population) would dominate the loss function.
- The scaler was fit on the training set only to prevent data leakage, then applied to both training and test sets.
- Decision Tree Regressor is invariant to scaling, but scaling was applied for consistency.

### Train-Test Split Configuration
- **80% training / 20% testing** split using `train_test_split` with `random_state=42` for reproducibility.
- Result: 16,512 training samples and 4,128 test samples.

### Models Selected and Rationale
1. **Linear Regression** — Ordinary Least Squares baseline. Establishes a lower bound for performance assuming a linear relationship.
2. **Ridge Regression** (`alpha=1.0`) — Linear model with L2 regularization. Reduces overfitting by penalizing large coefficients, often generalizing better than plain Linear Regression.
3. **Decision Tree Regressor** (`max_depth=5`) — Non-parametric model that captures non-linear patterns and feature interactions, providing a contrast to the linear models.

---

## Section 2: Results

### Performance Comparison Table

| Model | RMSE | R-squared |
|-------|------|-----------|
| Decision Tree | 0.7242 | 0.5997 |
| Ridge Regression | 0.7456 | 0.5758 |
| Linear Regression | 0.7456 | 0.5758 |

*(Values from notebook execution with `random_state=42`, `test_size=0.2`.)*

### Interpretation
- **RMSE (Root Mean Squared Error):** The Decision Tree achieves the lowest RMSE (0.7242), meaning its predictions deviate least from actual house prices. Both linear models show higher RMSE (0.7456), indicating ~3% larger average prediction errors.
- **R² (Coefficient of Determination):** The Decision Tree explains ~60% of the variance in house prices, outperforming both linear models (~57.6%).
- The performance gap between Ridge and Linear Regression is negligible, suggesting that L2 regularization with `alpha=1.0` provides marginal benefit on this dataset.

### Best Model
The **Decision Tree Regressor (max_depth=5)** is the best-performing model with the lowest RMSE and highest R². Its ability to capture non-linear relationships (e.g., interaction effects between income and location) gives it an edge over the linear models.

---

## Section 3: Conclusions

### Why the Best Model Outperformed Others
The Decision Tree outperformed both linear models because house prices exhibit **non-linear relationships** with features such as location (latitude/longitude) and income. For example, the marginal effect of income on house prices differs across geographic regions — a pattern a linear model cannot capture without explicit interaction terms. The Decision Tree automatically learns these splits.

### Real-World Implications
- **Median income (`MedInc`) is the strongest predictor** of house prices, confirmed by both correlation analysis and feature importance.
- **Geographic location** (latitude/longitude) significantly impacts price predictions, as expected in real estate.
- Tree-based models are a natural choice for housing price prediction problems where interactions between features are expected.

### Limitations & Potential Improvements
1. **Hyperparameter Tuning:** GridSearchCV or RandomizedSearchCV could optimize `max_depth`, `min_samples_split`, and `alpha` (Ridge) for better performance.
2. **Ensemble Methods:** Random Forest or Gradient Boosting (XGBoost/LightGBM) typically outperform single Decision Trees by reducing variance through averaging.
3. **Feature Engineering:** Polynomial features, log-transforms, and one-hot encoding of binned geographic coordinates could capture additional signal.
4. **Outlier Treatment:** The dataset contains high-value outliers that may distort both training and evaluation.
5. **Target Transformation:** A log-transform of the target variable could normalize its skewed distribution and improve model performance.

---

*Report generated from D:\project\maincraft\task-2\AI_ML_Task2_Model_Comparison.ipynb*

### How to Export to PDF
To convert this markdown report to PDF:
1. **Using VS Code:** Right-click the file → "Markdown PDF: Export (pdf)" with the Markdown PDF extension installed.
2. **Using pandoc:** Run `pandoc AI_ML_Task2_Report.md -o AI_ML_Task2_Report.pdf --pdf-engine=pdflatex`
3. **Using Typora:** Open the markdown file → File → Export → PDF.
4. **Using Jupyter:** Open the notebook → File → Download as → PDF via LaTeX (.pdf).
