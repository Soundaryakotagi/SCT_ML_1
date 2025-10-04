import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# ---- Load Dataset ----
df = pd.read_csv('https://raw.githubusercontent.com/Shitao-zz/Kaggle-House-Prices-Advanced-Regression-Techniques/refs/heads/master/input/train.csv')

# ---- Data Preparation ----
y = df['SalePrice']
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]

# ---- Data Splitting ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# ---- Linear Regression ----
lr = LinearRegression()
lr.fit(X_train, y_train)

y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# ---- Random Forest ----
rf = RandomForestRegressor(max_depth=5, random_state=100)
rf.fit(X_train, y_train)

y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

# ---- Model Comparison Table ----
df_models = pd.DataFrame([
    ['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2],
    ['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]
], columns=['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2'])

print("\n>> Model Comparison:")
print(df_models)

# ---- Save Best Model ----
best_model = rf if rf_test_r2 > lr_test_r2 else lr
joblib.dump(best_model, "best_house_price_model.pkl")
print("\n✅ Best model saved as best_house_price_model.pkl")

# ---- Data Visualization ----
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_rf_test_pred, color='blue', alpha=0.3, label='Random Forest Predictions (Test)')
plt.scatter(y_test, y_lr_test_pred, color='green', alpha=0.3, label='Linear Regression Predictions (Test)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Predictions vs Actual (Random Forest & Linear Regression)')
plt.legend()
plt.show()

