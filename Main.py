import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

# print("Let Ponyo be smarter!")

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_PATH_CSV = os.path.join(HOUSING_PATH, "housing.csv")


# Prepare Data
def create_data_path(housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)

# ------ Read Data ------
housing = pd.read_csv(HOUSING_PATH_CSV)

# ------ Take a Quick Look ------
# housing.info()
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())
#
# housing.hist(bins=50, figsize=(20, 15))
# plt.show()

# ------ Create a Test Set ------
# create test set manually
# def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]
#
# train_set, test_set = split_train_test(housing, 0.2)
# print(len(train_set), "train +", len(test_set), "testpip")
# def test_set_check(identifier, test_ratio, hash):
#     return hash(np.int64(identifier)).digest()

# create test data set with sklearn and remove cat
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# print(housing["income_cat"].value_counts()/len(housing))
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# ------ Discover and Visualize the Dta to Gain Insights ------
housing = strat_train_set.copy()
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#              s=housing["population"]/100, label="population", figsize=(10,7),
#              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
# plt.legend()
# plt.show()

# ------ Looking for Correlations ------
corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))
# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12,8))
# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.show()
