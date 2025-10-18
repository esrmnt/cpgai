import matplotlib.pyplot as plt
from numpy import mod
import pandas as pd
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)  # type: ignore[attr-defined]

print(fetch_california_housing()['DESCR'])


# print(X.shape, y.shape)
# print(X[:5])
# print(y[:5])

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsRegressor(n_neighbors=2))
])

model = GridSearchCV(estimator=pipeline,
             param_grid={'model__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, 
             cv=3)

model.fit(X, y)
print(pd.DataFrame(model.cv_results_))

Pipeline(steps=[('scaler', StandardScaler()), ('model', KNeighborsRegressor())])
predictions =  model.predict(X)

plt.scatter(predictions, y)
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.show()
