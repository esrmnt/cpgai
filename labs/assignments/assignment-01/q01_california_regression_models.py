import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from pandas.plotting import scatter_matrix
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Lasso, Ridge

housing = fetch_california_housing(as_frame=True)

# Explore the dataset
print("\nCalifornia Housing Dataset Info:")
print(housing.frame.info())
print(housing.frame.describe())

# Visualize the relationship between Median Income and Median House Value
housing.frame.plot(
    kind="scatter",
    x="MedInc",
    y="MedHouseVal",
    alpha=0.1
)
plt.show()

# Prepare features and target
X = housing.frame.drop('MedHouseVal', axis=1)
y = housing.frame['MedHouseVal']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform linear regression fit and train
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate the model
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"\nLinear Regression")
print(f"Train R2 Score: {train_score:.4f} test R2 Score: {test_score:.4f}")

# -------------------------------
# Lasso Regression
# -------------------------------

print("\nLasso Regression")

lasso_alphas = np.logspace(-6, 2, 30)
lasso_cost_results = {}
lasso_score_results = {}

for alpha in lasso_alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    
    lasso_cost_results[alpha] = np.sum((lasso.predict(X_train_scaled) - y_train) ** 2) + alpha * np.sum(np.abs(lasso.coef_))
    lasso_score_results[alpha] = lasso.score(X_test_scaled, y_test)

    print(f"Lasso alpha={alpha:15.10f} -> Train Cost: {lasso_cost_results[alpha]:10.4f} | Test R2: {lasso_score_results[alpha]:.4f}")

# Plot Lasso cost function vs alpha
plt.plot(list(lasso_cost_results.keys()), list(lasso_cost_results.values()), marker='x')
plt.xlabel('Alpha')
plt.ylabel('Cost Function Value')
plt.title('Lasso Regression Cost Function vs Alpha')
plt.xscale('log')
plt.show()

# Fit final Lasso model with best alpha
best_alpha_lasso = max(lasso_score_results, key=lasso_score_results.get)
lasso_final = Lasso(alpha=best_alpha_lasso, max_iter=10000)
lasso_final.fit(X_train_scaled, y_train)

# Predict on test set
lasso_prediction = lasso_final.predict(X_test_scaled)

# Create a DataFrame to compare actual vs predicted values
pred_df = pd.DataFrame({
    "Actuals": y_test.values,
    "Predicted": lasso_prediction
}, index=y_test.index)

print("\nPredictions (first 25 rows):")
print(pred_df.head(25))

mse = metrics.mean_squared_error(y_test, lasso_prediction)
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, lasso_prediction)
r2 = metrics.r2_score(y_test, lasso_prediction)

print(f"\nFinal Lasso (alpha={best_alpha_lasso:.10f}) results on TEST set Test R2: {r2:.4f} Test MSE: {mse:.4f} | Test RMSE: {rmse:.4f} | Test MAE: {mae:.4f}")


# -------------------------------
# Ridge Regression
# -------------------------------
alphas_ridge = np.logspace(-6, 6, 30)
ridge_cost_results = {}
ridge_score_results = {}

print("\nRidge results:")
for alpha in alphas_ridge:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    
    ridge_cost_results[alpha] = np.sum((ridge.predict(X_train_scaled) - y_train) ** 2) + alpha * np.sum(ridge.coef_ ** 2)
    ridge_score_results[alpha] = ridge.score(X_test_scaled, y_test)

    print(f"Ridge alpha={alpha:15.10f} -> Train Cost: {ridge_cost_results[alpha]:10.4f} | Test R2: {ridge_score_results[alpha]:.4f}")

# Plot Ridge cost function vs alpha
plt.plot(list(ridge_cost_results.keys()), list(ridge_cost_results.values()), marker='x')
plt.xlabel('Alpha')
plt.ylabel('Cost Function Value')
plt.title('Ridge Regression Cost Function vs Alpha')
plt.xscale('log')
plt.show()

# Fit final Ridge model with best alpha
best_alpha_ridge = max(ridge_score_results, key=ridge_score_results.get)
ridge_final = Ridge(alpha=best_alpha_ridge)
ridge_final.fit(X_train_scaled, y_train)

# Predict on test set
ridge_prediction = ridge_final.predict(X_test_scaled)

# Create a DataFrame to compare actual vs predicted values
pred_df_ridge = pd.DataFrame({
    "Actuals": y_test.values,
    "Predicted": ridge_prediction
}, index=y_test.index)

print("\nRidge Predictions (first 25 rows):")
print(pred_df_ridge.head(25))

mse_ridge = metrics.mean_squared_error(y_test, ridge_prediction)
rmse_ridge = np.sqrt(mse_ridge)
mae_ridge = metrics.mean_absolute_error(y_test, ridge_prediction)
r2_ridge = metrics.r2_score(y_test, ridge_prediction)

print(f"\nFinal Ridge (alpha={best_alpha_ridge:.10f}) results on TEST set Test R2: {r2_ridge:.4f} Test MSE: {mse_ridge:.4f} | Test RMSE: {rmse_ridge:.4f} | Test MAE: {mae_ridge:.4f}")