import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

data_root = "https://github.com/ageron/data/raw/main/"
lifestat = pd.read_csv(data_root + "lifesat/lifesat.csv")

print(lifestat.shape)

print(lifestat.head())

X = lifestat[["GDP per capita (USD)"]].values
y = lifestat["Life satisfaction"].values

lifestat.plot(kind="scatter", x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23500, 62500, 4, 9])
plt.show()

# model = LinearRegression()
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X, y)

X_new = [[37655.2]]
print("Predicted life satisfaction for GDP per capita of $37,655.2:", model.predict(X_new))