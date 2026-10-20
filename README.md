# Car Price Prediction with Machine Learning - CodeAlpha Task 3

## 📌 Project Overview
This project predicts car prices based on various features like year, mileage, fuel type, etc. using Machine Learning algorithms.

## 🎯 Objective
Build a model to estimate the selling price of used cars to help buyers and sellers.

## 📊 Dataset
- Source: Car Dekho Dataset (CodeAlpha provided)
- Features: Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-Learn (Linear Regression, Random Forest)
- Matplotlib, Seaborn

## 🤖 Models Used
1. **Linear Regression** - Baseline model
2. **Random Forest Regressor** - Best performance with 85%+ R2 Score

## 📈 Results
- Random Forest achieved higher accuracy
- Key factors: Present_Price, Year, Kms_Driven
- Visualization: Actual vs Predicted price, Feature Importance

## 🚀 How to Run
```bash
pip install pandas sklearn matplotlib seaborn
python Car_Price_Prediction.py
