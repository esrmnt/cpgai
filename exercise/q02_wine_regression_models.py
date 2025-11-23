import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine


wine = load_wine(as_frame=True)
# Explore the dataset
print("\nWine Quality Dataset Info:")
print(wine.frame.info())
print(wine.frame.describe())    
print(wine.frame.head())