import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ====== DATA LOADING ======
housing = fetch_california_housing(as_frame=True)
df = pd.concat([housing.data, housing.target], axis=1)
df.rename(columns={'MedHouseVal': 'HousePrice'}, inplace=True)
print("Data loaded:", df.shape)

# ====== PREPROCESSING ======
X = df.drop('HousePrice', axis=1)
y = df['HousePrice']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# ====== TRAIN/TEST SPLIT ======
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ====== MODEL TRAINING ======
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42)
}
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

# ====== EVALUATION ======
results = []
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.append({'Model': name, 'RMSE': rmse, 'R-squared': r2})
    print(f"{name:25s} | RMSE: {rmse:.4f} | R-squared: {r2:.4f}")

results_df = pd.DataFrame(results).sort_values('RMSE', ascending=True).reset_index(drop=True)
print("\nBest model:", results_df.iloc[0]['Model'])
print("Best RMSE:", results_df.iloc[0]['RMSE'])
print("Best R-squared:", results_df.iloc[0]['R-squared'])

# ====== SAVE MODEL ======
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
joblib.dump(best_model, './best_housing_model.pkl')
print(f"\nModel saved: best_housing_model.pkl")

# ====== PLOTS ======
y_pred_best = best_model.predict(X_test)
residuals = y_test - y_pred_best

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test, y_pred_best, alpha=0.5, edgecolors='none', s=20)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Prices')
axes[0].set_ylabel('Predicted Prices')
axes[0].set_title(f'Actual vs Predicted House Prices ({best_model_name})')
axes[0].legend()
axes[0].axis('equal')

# Plot 2: Performance Comparison
x_pos = np.arange(len(results_df))
width = 0.35
ax1 = axes[1]
ax1_twin = ax1.twinx()
ax1.bar(x_pos - width/2, results_df['RMSE'], width, label='RMSE', color='steelblue', edgecolor='black')
ax1_twin.bar(x_pos + width/2, results_df['R-squared'], width, label='R-squared', color='coral', edgecolor='black')
ax1.set_xlabel('Model')
ax1.set_ylabel('RMSE', color='steelblue')
ax1_twin.set_ylabel('R-squared', color='coral')
ax1.set_title('Model Performance Comparison')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(results_df['Model'], rotation=15, ha='right')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Plot 3: Residual Distribution
axes[2].hist(residuals, bins=40, density=True, alpha=0.6, color='steelblue', edgecolor='black', label='Residuals')
sns.kdeplot(residuals, ax=axes[2], color='darkred', linewidth=2, label='KDE')
axes[2].axvline(x=0, color='black', linestyle='--', linewidth=1, label='Zero Error')
axes[2].set_xlabel('Residual (Actual - Predicted)')
axes[2].set_ylabel('Density')
axes[2].set_title(f'Residual Distribution ({best_model_name})')
axes[2].legend()

plt.tight_layout()
plt.savefig('performance_plots.png', dpi=150)
plt.show()

print("\nPipeline completed successfully!")
print("Output files: best_housing_model.pkl, performance_plots.png")
