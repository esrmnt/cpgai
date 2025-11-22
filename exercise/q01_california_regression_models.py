import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression, LinearRegression, Ridge
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)

# print(housing.frame.info())
# print(housing.frame.describe())

# housing.frame.plot(
#     kind="scatter",
#     x="MedInc",
#     y="MedHouseVal",
#     alpha=0.1
# )
# plt.show()

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

print(f"Train R² Score: {train_score:.4f} test R² Score: {test_score:.4f}")

# -------------------------------
# Lasso Regression
# -------------------------------

lasso_alphas = np.logspace(-6, 2, 30)
lasso_results = {}

for alpha in lasso_alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    
    score = lasso.score(X_test_scaled, y_test)
    cost_function = np.sum((lasso.predict(X_train_scaled) - y_train) ** 2) + alpha * np.sum(np.abs(lasso.coef_))

    lasso_results[alpha] = {
        "cost_function": cost_function, 
        "score": score
    }

best_alpha_from_results = max(lasso_results.items(), key=lambda kv: kv[1]["score"])[0]
best_score_from_results = lasso_results[best_alpha_from_results]["score"]
print(f"Best alpha from lasso_results: {best_alpha_from_results} -> test R²: {best_score_from_results:.4f}")

plt.plot(list(lasso_results.keys()), [v["cost_function"] for v in lasso_results.values()], marker='x')
plt.xlabel('Alpha')
plt.ylabel('Cost Function Value')
plt.title('Lasso Regression Cost Function vs Alpha')
plt.show()


# -------------------------------
# Ridge Regression
# -------------------------------
alphas_ridge = [0.001, 0.01, 0.1, 1, 10, 50, 100]
best_alpha_ridge = None
best_score_ridge = -np.inf

print("\nRidge results:")
for a in alphas_ridge:
    r = Ridge(alpha=a)
    r.fit(X_train_scaled, y_train)
    tr = r.score(X_train_scaled, y_train)
    te = r.score(X_test_scaled, y_test)
    print(f" alpha={a:>6} -> Train R²: {tr:.4f} | Test R²: {te:.4f}")
    if te > best_score_ridge:
        best_score_ridge = te
        best_alpha_ridge = a

print(f"Best Ridge alpha: {best_alpha_ridge} with Test R²: {best_score_ridge:.4f}")

# Optionally, fit final Ridge model with best alpha
final_ridge = Ridge(alpha=best_alpha_ridge)
final_ridge.fit(X_train_scaled, y_train)
print("\nFinal Ridge coefficients (aligned with feature names):")
coef_df = pd.Series(final_ridge.coef_, index=X.columns).sort_values(key=abs, ascending=False)
print(coef_df)