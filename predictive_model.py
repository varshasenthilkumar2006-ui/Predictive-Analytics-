# ======================================================
# Predictive Analytics Project
# Predict Tip amount using Total Bill (Linear Regression)
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# -------------------------------
# 1. Load Data
# -------------------------------
df = sns.load_dataset('tips')

print("===== Dataset Shape =====")
print(df.shape)
print("\nFirst 5 rows:")
print(df.head())

# -------------------------------
# 2. Select Features
# -------------------------------
X = df[['total_bill']]          # Input
y = df['tip']                   # What we want to predict

# -------------------------------
# 3. Split Data (Train & Test)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining size:", X_train.shape[0])
print("Testing size :", X_test.shape[0])

# -------------------------------
# 4. Build the Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 5. Make Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 6. Evaluate the Model
# -------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== Model Performance =====")
print(f"Mean Squared Error : {mse:.3f}")
print(f"R² Score           : {r2:.3f}")

print(f"\nModel Equation: Tip = {model.coef_[0]:.3f} * Total_Bill + {model.intercept_:.3f}")

# -------------------------------
# 7. Visualization
# -------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Tip', alpha=0.6)
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Tip')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.title('Actual vs Predicted Tip')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# 8. Predict for New Value
# -------------------------------
new_bill = np.array([[50]])  # Example: $50 bill
predicted_tip = model.predict(new_bill)

print(f"\nFor a Total Bill of $50, Predicted Tip = ${predicted_tip[0]:.2f}")

print("\n===== Predictive Analytics Project Completed! =====")