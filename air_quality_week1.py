import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("air_quality.csv")

cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]
df = df[cols].copy()

df = df.fillna(df.mean(numeric_only=True))

X = df[["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]]
y = df["AQI"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_pred = lin_model.predict(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n--- Model Evaluation ---")
print("Linear Regression:")
print("  MSE:", round(mean_squared_error(y_test, lin_pred), 2))
print("  R2 :", round(r2_score(y_test, lin_pred), 3))

print("\nRandom Forest:")
print("  MSE:", round(mean_squared_error(y_test, rf_pred), 2))
print("  R2 :", round(r2_score(y_test, rf_pred), 3))

importances = rf_model.feature_importances_
print("\n--- Feature Importances ---")
for feat, imp in zip(X.columns, importances):
    print(f"{feat}: {imp:.3f}")

plt.figure(figsize=(6,6))
plt.scatter(y_test, rf_pred, alpha=0.6, label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label="Perfect Fit")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Random Forest: Actual vs Predicted AQI")
plt.legend()
plt.grid(True)
plt.show()
