import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer

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
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
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
# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))
# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributesØ], figsize=(12,8))
# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.show()

# ------ Experimenting with Attribute Combinations ------
# housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# housing["population_per_household"] = housing["population"]/housing["households"]
# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

# ------ Prepare the Data for ML Algorithms ------
# ------ Data cleaning ------
housing = strat_train_set.drop("median_house_value", axis=1)
# housing.info()
housing_labels = strat_train_set["median_house_value"].copy()
# print(len(housing_labels))
# housing.dropna(subset=["total_bedrooms"]) ## Get red of the corresponding districts
# housing.drop("total_bedrooms", axis=1) ## Get rid of the whole attribute
# median = housing["total_bedrooms"].median() ## Set the values to some value(zero, the mean, the median, etc...)
# ------ Imputer ------
# imputer = Imputer(strategy="median") # most_frequent
housing_num = housing.drop("ocean_proximity", axis=1)
# imputer.fit(housing_num)
# print(imputer.statistics_)
# print(housing_num.median().values)

# ------ Handling Text and Categorical Attributes ------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# # ------ One-hot encoder ------
# encoder = LabelEncoder()
# housing_cat = housing["ocean_proximity"]
# housing_cat_encoded = encoder.fit_transform(housing_cat)
# print(housing_cat_encoded)
# print(encoder.classes_)
# # ------ One-hot encoder ------
# onehot = OneHotEncoder()
# housing_cat_1hot = onehot.fit_transform(housing_cat_encoded.reshape(-1, 1))
# print(housing_cat_1hot.toarray())
# ------ LabelBinarizer ------
# labelBinarizer = LabelBinarizer() #labelBinarizer = LabelBinarizer(sparse_output=True) -> Get sparse matrix
# housing_cat_1hot = labelBinarizer.fit_transform(housing_cat)
# print(housing_cat_1hot)

# ------ Custom Transformers ------
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
# housing_extra_attribs = attr_adder.transform(housing.values)

# ------ Transformation Pipelines ------
# @param using pip3 install scikit-learn==0.18.0
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# num_pipeline = Pipeline([('imputer', Imputer(strategy="median")),
#                          ('attribs_adder', CombinedAttributesAdder()),
#                          ('std_scaler', StandardScaler())])
# housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.base import BaseEstimator, TransformerMixin

# Create a class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizer()),
    ])

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared)
# print(housing_prepared.shape)

# ------ Training ------
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
# print("Predictions:", lin_reg.predict(some_data_prepared))
# print("Labels", list(some_labels))

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)