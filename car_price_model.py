# ============================================================
# CAR PRICE PREDICTION - MACHINE LEARNING PROJECT
#============================================================

# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Skipping XGBoost model.")

# Create output folder for plots
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("  CAR PRICE PREDICTION - ML PROJECT")
print("=" * 60)


#DATA Collection
print("\n[STEP 2] Loading dataset...")


df = pd.read_csv("CarPrice_Assignment.csv")

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")


#Data Understanding
print("\n[STEP 3] Understanding the data...")

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Data Types ---")
print(df.dtypes)

# ---- Graph 1: Price Distribution ----
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(df['price'], bins=30, color='steelblue', edgecolor='white')
plt.title('Car Price Distribution')
plt.xlabel('Price (USD)')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist(np.log1p(df['price']), bins=30, color='darkorange', edgecolor='white')
plt.title('Log-Transformed Price Distribution')
plt.xlabel('Log(Price)')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig("outputs/price_distribution.png", dpi=150)
plt.show()
print("Saved: outputs/price_distribution.png")

# ---- Graph 2: Correlation Heatmap ----
plt.figure(figsize=(14, 10))
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5, annot_kws={"size": 8})
plt.title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.show()
print("Saved: outputs/correlation_heatmap.png")

# ---- Graph 3: Top features correlated with price ----
price_corr = corr_matrix['price'].drop('price').sort_values(ascending=False)
plt.figure(figsize=(10, 6))
price_corr.plot(kind='bar', color=['green' if v > 0 else 'red' for v in price_corr])
plt.title('Feature Correlation with Price')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("outputs/feature_price_correlation.png", dpi=150)
plt.show()
print("Saved: outputs/feature_price_correlation.png")

# ---- Graph 4: Fuel Type vs Price Boxplot ----
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
df.boxplot(column='price', by='fueltype', figsize=(5, 4))
plt.title('Fuel Type vs Price')
plt.suptitle('')

plt.subplot(1, 2, 2)
df.boxplot(column='price', by='carbody')
plt.title('Car Body vs Price')
plt.suptitle('')
plt.xticks(rotation=30, ha='right')

plt.tight_layout()
plt.savefig("outputs/category_vs_price.png", dpi=150)
plt.show()
print("Saved: outputs/category_vs_price.png")


# ============================================================
# STEP 4: DATA PREPROCESSING
# ============================================================
print("\n[STEP 4] Preprocessing data...")

# Make a copy so original data is safe
data = df.copy()

# 4.1 - Extract car company name from CarName column
# The CarName column has format: "toyota corolla" -> we want "toyota"
data['CarCompany'] = data['CarName'].apply(lambda x: x.split(' ')[0].lower())

# Fix common typos/inconsistencies in company names
company_fixes = {
    'vokswagen': 'volkswagen',
    'vw': 'volkswagen',
    'toyouta': 'toyota',
    'maxda': 'mazda',
    'Nissan': 'nissan',
    'porcshce': 'porsche'
}
data['CarCompany'] = data['CarCompany'].replace(company_fixes)

print(f"Unique car companies: {sorted(data['CarCompany'].unique())}")

# 4.2 - Handle missing values (this dataset has none, but we check anyway)
print(f"\nMissing values: {data.isnull().sum().sum()}")

# 4.3 - Remove duplicates
before = len(data)
data.drop_duplicates(inplace=True)
print(f"Removed {before - len(data)} duplicate rows")


# car_ID is just an index, CarName is replaced by CarCompany
data.drop(['car_ID', 'CarName'], axis=1, inplace=True)

print(f"\nData shape after cleaning: {data.shape}")
print(f"Remaining columns: {list(data.columns)}")



# FEATURE ENGINEERING

print("\n[STEP 5] Feature Engineering...")

# 5.1 - Create new useful features
# Power-to-weight ratio (high value = sporty/fast car = higher price)
data['power_to_weight'] = data['horsepower'] / data['curbweight']

# Engine displacement per cylinder
data['displacement_per_cyl'] = data['enginesize'] / data['cylindernumber'].replace({
    'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'eight': 8, 'twelve': 12
}).astype(float)

# Fuel efficiency (combined city + highway)
data['avg_mpg'] = (data['citympg'] + data['highwaympg']) / 2

print("New features created: power_to_weight, displacement_per_cyl, avg_mpg")

# 5.2 - Encode categorical columns
# Identify categorical columns
cat_cols = data.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns to encode: {cat_cols}")

# Label Encoding for categorical features
le = LabelEncoder()
label_encoders = {}  # Store encoders for deployment use

for col in cat_cols:
    le_temp = LabelEncoder()
    data[col] = le_temp.fit_transform(data[col])
    label_encoders[col] = le_temp

print("All categorical columns encoded successfully.")

# 5.3 - Separate features and target variable
X = data.drop('price', axis=1)
y = data['price']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# 5.4 - Feature Scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("Feature scaling applied (StandardScaler)")

# 5.5 - Feature Importance (using a quick Random Forest)
print("\nRunning preliminary feature importance check...")
temp_rf = RandomForestRegressor(n_estimators=50, random_state=42)
temp_rf.fit(X_scaled, y)

feat_importance = pd.Series(temp_rf.feature_importances_, index=X.columns)
feat_importance = feat_importance.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
feat_importance.head(15).plot(kind='bar', color='steelblue', edgecolor='white')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.show()
print("Saved: outputs/feature_importance.png")

# Select top features (importance > 0.01)
selected_features = feat_importance[feat_importance > 0.01].index.tolist()
print(f"\nSelected {len(selected_features)} features: {selected_features}")

X_selected = X_scaled[selected_features]



# STEP 6: MODEL BUILDING

print("\n[STEP 6] Building ML Models...")

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")

# Define models to compare
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

print(f"\nModels to train: {list(models.keys())}")



# STEP 7: MODEL EVALUATION

print("\n[STEP 7] Evaluating Models...")

results = {}

for name, model in models.items():
    print(f"\n  Training: {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv_r2 = cross_val_score(model, X_selected, y, cv=5, scoring='r2').mean()

    results[name] = {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2 Score': round(r2, 4),
        'CV R2 Score': round(cv_r2, 4),
        'model': model,
        'predictions': y_pred
    }

    print(f"    MAE:        ${mae:,.2f}")
    print(f"    RMSE:       ${rmse:,.2f}")
    print(f"    R² Score:   {r2:.4f} ({r2*100:.1f}%)")
    print(f"    CV R² Score:{cv_r2:.4f}")

# Summary Table
print("\n--- Model Comparison Summary ---")
summary = pd.DataFrame({
    k: {
        'MAE': v['MAE'],
        'RMSE': v['RMSE'],
        'R2': v['R2 Score'],
        'CV R2': v['CV R2 Score']
    }
    for k, v in results.items()
}).T
print(summary)

# ---- Graph: Model Comparison Bar Chart ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
model_names = list(results.keys())
colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']

# R2 Score
axes[0].bar(model_names, [results[m]['R2 Score'] for m in model_names], color=colors)
axes[0].set_title('R² Score (Higher is Better)')
axes[0].set_ylabel('R² Score')
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis='x', rotation=30)

# MAE
axes[1].bar(model_names, [results[m]['MAE'] for m in model_names], color=colors)
axes[1].set_title('MAE (Lower is Better)')
axes[1].set_ylabel('MAE (USD)')
axes[1].tick_params(axis='x', rotation=30)

# RMSE
axes[2].bar(model_names, [results[m]['RMSE'] for m in model_names], color=colors)
axes[2].set_title('RMSE (Lower is Better)')
axes[2].set_ylabel('RMSE (USD)')
axes[2].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)
plt.show()
print("Saved: outputs/model_comparison.png")

# ---- Graph: Actual vs Predicted (Best Model) ----
best_model_name = max(results, key=lambda x: results[x]['R2 Score'])
best_preds = results[best_model_name]['predictions']

plt.figure(figsize=(8, 6))
plt.scatter(y_test, best_preds, alpha=0.6, color='steelblue', edgecolors='white', s=60)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title(f'Actual vs Predicted - {best_model_name}')
plt.legend()
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png", dpi=150)
plt.show()
print(f"Saved: outputs/actual_vs_predicted.png")

print(f"\n>>> Best Model: {best_model_name} (R² = {results[best_model_name]['R2 Score']})")


#MODEL OPTIMIZATION (Hyperparameter Tuning)

print("\n[STEP 8] Optimizing Best Model...")

best_model_obj = results[best_model_name]['model']

# Tune Random Forest if it's the best (most common case)
if "Random Forest" in best_model_name:
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
elif "Gradient Boosting" in best_model_name:
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    grid_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
else:
    # For Linear Regression, no tuning needed; just report
    print(f"No hyperparameter tuning needed for {best_model_name}")
    final_model = best_model_obj
    grid_search = None

if grid_search is not None:
    print("Running GridSearchCV (this may take a moment)...")
    grid_search.fit(X_train, y_train)
    final_model = grid_search.best_estimator_

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Score:   {grid_search.best_score_:.4f}")

    y_pred_tuned = final_model.predict(X_test)
    r2_tuned = r2_score(y_test, y_pred_tuned)
    mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
    rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))

    print(f"\n--- After Optimization ---")
    print(f"  R² Score: {r2_tuned:.4f} ({r2_tuned*100:.1f}%)")
    print(f"  MAE:      ${mae_tuned:,.2f}")
    print(f"  RMSE:     ${rmse_tuned:,.2f}")
else:
    final_model = best_model_obj


#SAVE MODEL FOR DEPLOYMENT

print("\n[STEP 9] Saving model artifacts...")

# Save the final model
joblib.dump(final_model, "outputs/car_price_model.pkl")
print("Saved: outputs/car_price_model.pkl")

# Save the scaler
joblib.dump(scaler, "outputs/scaler.pkl")
print("Saved: outputs/scaler.pkl")

# Save the label encoders
joblib.dump(label_encoders, "outputs/label_encoders.pkl")
print("Saved: outputs/label_encoders.pkl")

# Save selected features list
joblib.dump(selected_features, "outputs/selected_features.pkl")
print("Saved: outputs/selected_features.pkl")

# Save all column names (needed for deployment)
all_feature_cols = list(X.columns)
joblib.dump(all_feature_cols, "outputs/all_feature_cols.pkl")
print("Saved: outputs/all_feature_cols.pkl")


#FINAL OUTPUT SUMMARY

print("\n" + "=" * 60)
print("  FINAL PROJECT SUMMARY")
print("=" * 60)
print(f"  Dataset size:    {df.shape[0]} rows, {df.shape[1]} features")
print(f"  Features used:   {len(selected_features)}")
print(f"  Best model:      {best_model_name}")
print(f"  R² Score:        {results[best_model_name]['R2 Score']} ({results[best_model_name]['R2 Score']*100:.1f}%)")
print(f"  MAE:             ${results[best_model_name]['MAE']:,.2f}")
print(f"  RMSE:            ${results[best_model_name]['RMSE']:,.2f}")
print(f"\n  Output files in: ./outputs/")
print(f"    - car_price_model.pkl")
print(f"    - scaler.pkl")
print(f"    - label_encoders.pkl")
print(f"    - selected_features.pkl")
print(f"    - all_feature_cols.pkl")
print(f"    - price_distribution.png")
print(f"    - correlation_heatmap.png")
print(f"    - feature_importance.png")
print(f"    - model_comparison.png")
print(f"    - actual_vs_predicted.png")
print("=" * 60)
print("\nProject completed! Run app.py to launch the web UI.")