import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import hashlib
import sklearn.model_selection

# print("Let Ponyo be smarter!")

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_PATH_CSV = os.path.join(HOUSING_PATH, "housing.csv")


# Prepare Data
def create_data_path(housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)


# start
# prepare_housing_data()

# Read Data
housing = pd.read_csv(HOUSING_PATH_CSV)


housing.info()
# housing["longitude"].value_counts()

# housing.hist(bins=50, figsize=(20, 15))
# plt.show()


# Create a Test Set
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# train_set, test_set = split_train_test(housing, 0.2)
# print(len(train_set), "train +", len(test_set), "testpip")
#
#
# def test_set_check(identifier, test_ratio, hash):
#     return hash(np.int64(identifier)).digest()

# Use Scikit-Learn to generate train test data
train_set, test_set = sklearn.model_selection.train_test_split(housing, test_size=0.2, random_state=42)

print(len(train_set), len(test_set))

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

count = housing["income_cat"].value_counts() / len(housing)
print(count)
