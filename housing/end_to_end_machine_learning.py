"""
End-to-end Machine Learning project using California Housing dataset.
"""

import tarfile
import numpy as np
import pandas as pd
import seaborn as sns
import urllib.request
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats

def load_housing_data():
    """Download and load the California housing dataset."""
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

def split_train_test(data, test_ratio):
    """Split the data into training and test sets."""
    return train_test_split(data, test_size=test_ratio, random_state=42)

def prepare_data(housing):
    """Prepare the data for training."""
    housing_num = housing.select_dtypes(include=[np.number])
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    # Create preprocessing pipelines
    num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )

    preprocessing = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return preprocessing

def train_model(preprocessing, X_train, y_train):
    """Train the model using RandomForestRegressor."""
    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using RMSE."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return rmse

def main():
    # Load data
    housing = load_housing_data()
    print("Data loaded successfully.")

    print(housing.shape)
    print(housing.info())
    print(housing["ocean_proximity"].value_counts())
    print(housing.describe())
  
    # Split the data
    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(housing, housing_labels, test_size=0.2, random_state=42)
    print("Data split into training and test sets.")

    # Prepare the data
    preprocessing = prepare_data(housing)
    print("Data preprocessing pipeline created.")

    # Train the model
    model = train_model(preprocessing, X_train, y_train)
    print("Model trained successfully.")

    # Evaluate the model
    rmse = evaluate_model(model, X_test, y_test)
    print(f"Model RMSE: {rmse:.2f}")

    # Cross-validation
    scores = cross_val_score(model, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=5)
    cv_rmse = -scores
    print("\nCross-validation scores:")
    print(f"Mean RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std() * 2:.2f})")

    # Save the model
    import joblib
    joblib.dump(model, "california_housing_model.pkl")
    print("\nModel saved as 'california_housing_model.pkl'")

if __name__ == "__main__":
    main()