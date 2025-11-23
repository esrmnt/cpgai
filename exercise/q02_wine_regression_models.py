import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine


wine = load_wine(as_frame=True)

# ----------------------------
# BASIC STRUCTURE
# ----------------------------
print("Shape:", wine.frame.shape)
print("\nDtypes:"), 
print(wine.frame.dtypes)
print("\nInfo:")
print(wine.frame.info())

print("\nHead:")
print(wine.frame.head())

print("\nRandom Sample:")
print(wine.frame.sample(5))

#----------------------------
#MISSING VALUES
#----------------------------

print("\nMissing Values in data:")

if wine.frame.isna().sum().sum() > 0:
    print("\nRows with Missing Values:")
    print(wine.frame[wine.frame.isna().any(axis=1)])

    plt.figure(figsize=(10,4))
    sns.heatmap(wine.frame.isna(), cbar=False)
    plt.title("Missing Value Map")
    plt.show()
elif wine.frame.isna().sum().sum() == 0:
    print("\nNo missing values found.")

# ----------------------------
# DESCRIPTIVE STATISTICS
# ----------------------------
print("\nDescribe:")
print(wine.frame.describe())


# ----------------------------
# UNIQUE VALUE CHECK
# ----------------------------
print("\nUnique Values per Column:")
for i, col in enumerate(wine.frame.columns, 1):
    print(f"{i:>2}. {col}: {wine.frame[col].nunique()} unique values")

# ----------------------------
# DISTRIBUTIONS
# ----------------------------
print("\nDistributions of data in the numeric columns:")
numeric_cols = wine.frame.select_dtypes(include=[np.number]).columns

wine.frame[numeric_cols].hist(figsize=(14, 10), bins=30)
plt.suptitle("Histograms")
plt.show()

# KDE example for one column (if needed)
# sns.kdeplot(wine.frame["alcalinity_of_ash"]); plt.show()

# ----------------------------
# CORRELATIONS
# ----------------------------
print("\nCorrelation Matrix:")
corr = wine.frame.corr(numeric_only=True)
print(corr)

plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------
# OUTLIERS (BOXPLOTS FOR NUMERICAL FEATURES)
# ----------------------------
plt.figure(figsize=(12, 6))
wine.frame[numeric_cols].boxplot(rot=90)
plt.title("Boxplots")
plt.show()

# ----------------------------
# TARGET RELATIONSHIPS
# ----------------------------
selected_cols = ["alcohol", "malic_acid", "color_intensity", "hue"]
target = "target"

for col in selected_cols:
    if col == target:
        continue
    plt.figure(figsize=(5, 4))
    plt.scatter(wine.frame[col], wine.frame[target], alpha=0.5)
    plt.xlabel(col)
    plt.ylabel(target)
    plt.title(f"{col} vs {target}")
    plt.show()

