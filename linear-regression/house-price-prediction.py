import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('data.csv')
print(df.head())
df = df.dropna()

X = df[['area', 'bedrooms', 'bathrooms']] 
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

plt.scatter(X_test['area'], y_test, color='blue', label='Actual Price')
plt.plot(X_test['area'], y_pred, color='red', label='Regression Line')
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (in ₹)")
plt.title("Simple Linear Regression: Area vs Price")
plt.show()