import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

wine = load_wine(as_frame=True)

enable_eda = False
enable_logistic_regression = True
enable_smb_rbf = True
enable_decision_tree = True

if enable_eda == True:  
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

# Prepare features and target
X = wine.frame.drop('target', axis=1)
y = wine.frame['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if enable_logistic_regression == True:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nLogistic Regression Model Accuracy: {accuracy:.4f}")

    metrics_cnf = metrics.confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(metrics_cnf)

    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1 Score: {f1:.4f}")


if enable_smb_rbf == True:
    model = SVC(kernel='rbf', gamma='scale')
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nSVM RBF Model Accuracy: {accuracy:.4f}")

    metrics_cnf = metrics.confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(metrics_cnf)

    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1 Score: {f1:.4f}")

if enable_decision_tree == True:
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nDecision Tree Model Accuracy: {accuracy:.4f}")

    metrics_cnf = metrics.confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(metrics_cnf)

    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1 Score: {f1:.4f}")