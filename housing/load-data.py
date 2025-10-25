from pathlib import Path
import pandas as pd 
import numpy as np
import tarfile
import urllib.request
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

def load_housing_data(housing_url: str, housing_path: Path) -> pd.DataFrame:
    tarball_path = housing_path / "housing.tgz"    

    if not tarball_path.is_file():
         housing_path.mkdir(parents=True, exist_ok=True)
         urllib.request.urlretrieve(housing_url, tarball_path)

    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path=housing_path)
    
    csv_path = housing_path / "housing" / "housing.csv"
    return pd.read_csv(csv_path)

def shuffle_and_split_data(data: pd.DataFrame, test_ratio: float = 0.2):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.loc[train_indices], data.loc[test_indices]

def plot_raw_housing_data(housing):
    housing.hist(bins=50, figsize=(20,15))
    plt.show()

if __name__ == "__main__":
    housing_url = "https://github.com/ageron/data/raw/main/housing.tgz"
    housing_path = Path("datasets")
    housing = load_housing_data(housing_url, housing_path)
    
    # housing["id"] = housing["longitude"] * 1000 + housing["latitude"]

    print(housing.head())
    print(housing.info())

    # Create income categories for stratification
    housing["income_cat"] = pd.cut(housing["median_income"],
                                 bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                 labels=[1, 2, 3, 4, 5])
    
    # Perform stratified split based on income categories
    stratified_train_set, stratified_test_set = train_test_split(
        housing, 
        test_size=0.2, 
        stratify=housing["income_cat"], 
        random_state=42
    )
    
    # Remove the income_cat column
    for set_ in (stratified_train_set, stratified_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
        
    print(f"Train set size: {len(stratified_train_set)}")
    print(f"Test set size: {len(stratified_test_set)}")

    print(stratified_train_set.head())
    print(stratified_train_set.info())

    housing.plot(
        kind="scatter", x="longitude", y="latitude", 
        alpha=0.2, grid=True,
        s=housing["population"]/100, label ="population",
        c="median_house_value", cmap ="jet", colorbar=True,
        legend=True, sharex=False, figsize=(10,7)
    )
    #plt.show()

    housing.plot(
        kind="scatter", x="median_income", y="median_house_value", alpha=0.4, grid=True
    )
    #plt.show()

    # median_bedroom = stratified_train_set["total_bedrooms"].median()
    # print(f"Median bedroom: {median_bedroom}")
    # stratified_train_set.fillna({"total_bedrooms" : median_bedroom}, inplace=True)
    # print(stratified_train_set.info())
    housing_cat = housing[["ocean_proximity"]]
    
    categorical_encoder = OneHotEncoder()
    housing_cat_1hot = categorical_encoder.fit_transform(housing_cat)
    print(housing_cat_1hot.toarray())