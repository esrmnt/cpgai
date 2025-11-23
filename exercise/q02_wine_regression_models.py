import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display
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
wine.frame.info()

print("\nHead:")
display(wine.frame.head())

print("\nRandom Sample:")
display(wine.frame.sample(5))

# ----------------------------
# MISSING VALUES
# ----------------------------
# print("\nMissing Values:\n", wine.frame.isna().sum())

# plt.figure(figsize=(10,4))
# sns.heatmap(wine.frame.isna(), cbar=False)
# plt.title("Missing Value Map")
# plt.show()

# # ----------------------------
# # DESCRIPTIVE STATISTICS
# # ----------------------------
# print("\nDescribe:")
# print(wine.frame.describe())


# # ----------------------------
# # UNIQUE VALUE CHECK
# # ----------------------------
# for col in wine.frame.columns:
#     print(f"{col}: {wine.frame[col].nunique()} unique values")

# # ----------------------------
# # DISTRIBUTIONS
# # ----------------------------
# numeric_cols = wine.frame.select_dtypes(include=[np.number]).columns

# wine.frame[numeric_cols].hist(figsize=(14, 10), bins=30)
# plt.suptitle("Histograms")
# plt.show()

# # KDE example for one column (if needed)
# # sns.kdeplot(wine.frame["colname"]); plt.show()

# # ----------------------------
# # CORRELATIONS
# # ----------------------------
# corr = wine.frame.corr(numeric_only=True)

# plt.figure(figsize=(12, 8))
# sns.heatmap(corr, cmap="coolwarm", annot=False)
# plt.title("Correlation Heatmap")
# plt.show()

# # ----------------------------
# # OUTLIERS (BOXPLOTS FOR NUMERICAL FEATURES)
# # ----------------------------
# plt.figure(figsize=(12, 6))
# wine.frame[numeric_cols].boxplot(rot=90)
# plt.title("Boxplots")
# plt.show()

# # ----------------------------
# # TARGET RELATIONSHIPS
# # ----------------------------
# target = "target"

# for col in numeric_cols:
#     if col == target:
#         continue
#     plt.figure(figsize=(5, 4))
#     plt.scatter(wine.frame[col], wine.frame[target], alpha=0.5)
#     plt.xlabel(col)
#     plt.ylabel(target)
#     plt.title(f"{col} vs {target}")
#     plt.show()

# # For categoricals → group mean
# cat_cols = wine.frame.select_dtypes(include="object").columns
# for col in cat_cols:
#     print(f"\nGroup mean for {col}:")
#     display(wine.frame.groupby(col)[target].mean().sort_values())

