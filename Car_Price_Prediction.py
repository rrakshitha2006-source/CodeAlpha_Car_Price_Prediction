# CodeAlpha Internship - Task 3: Car Price Prediction
# Author: Rakshitha R

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

print("=== Car Price Prediction Project ===")

# Load Dataset
# Download dataset from CodeAlpha link and upload as car_data.csv
df = pd.read_csv('car data.csv')
print(f"Dataset Shape: {df.shape}")
print(df.head())

# Data Cleaning
df.dropna(inplace=True)

# Label Encoding for categorical data
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    if col not in ['Car_Name']:
        df[col] = le.fit_transform(df[col].astype(str))

# Features and Target
# Assuming Selling_Price is target
target = 'Selling_Price' if 'Selling_Price' in df.columns else df.columns[-1]
X = df.drop([target, 'Car_Name'], axis=1, errors='ignore')
y = df[target]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1 - Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
print(f"\nLinear Regression R2: {r2_score(y_test, pred_lr):.4f}")

# Model 2 - Random Forest (Best)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
print(f"Random Forest R2: {r2_score(y_test, pred_rf):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_rf)):.2f}")

# Visualization
plt.figure(figsize=(8,5))
plt.scatter(y_test, pred_rf, alpha=0.5, color='blue')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()

# Feature Importance
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
feat_imp.plot(kind='barh')
plt.title("Feature Importance")
plt.show()

print("\nModel Ready for Prediction!")
