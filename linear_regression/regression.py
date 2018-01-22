# Analysis goal: predict housing prices in California

import os
import sys
import tarfile
from six.moves import urllib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAVE_PLOTS = False

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

print('Download the data')
DOWNLOAD_ROOT = 'https://github.com/ageron/handson-ml/raw/master/'
HOUSING_PATH = 'datasets/housing'
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + '/housing.tgz'


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    if os.path.isfile(tgz_path):
        print('File already present, skip download!')
        return
    print('Downloading from', housing_url)
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

print('Load the data')


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

housing = load_housing_data()
print(housing.head())  # shows first few entries
print(housing.info())  # shows column data types and general data frame info
print(housing['ocean_proximity'].value_counts())  # shows value abundance
print(housing.describe())  # shows count, mean, std etc for every column

if SAVE_PLOTS:
    print('Plot the data')
    plt.figure()
    housing.hist(bins=50, figsize=(20, 15))
    plt.savefig('housing.pdf')

# -----------------------------------------------------------------------------
# DATA SPLITTING
# -----------------------------------------------------------------------------

print('Split into train and test data set')

split_strategy = 'stratified'

if split_strategy == 'naive':
    def split_train_test(data, test_ratio):
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return data.iloc[train_indices], data.iloc[test_indices]

    train_set, test_set = split_train_test(housing, 0.2)

elif split_strategy == 'sklearn_naive':
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

elif split_strategy == 'hash_ID' or split_strategy == 'hash_ID_homemade':
    import hashlib

    def test_set_check(identifier, test_ratio, hash_):
        return hash_(np.int64(identifier)).digest()[-1] < 256 * test_ratio

    def split_train_test_by_id(data, test_ratio, id_column, hash_=hashlib.md5):
        ids = data[id_column]
        in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash_))
        return data.loc[~in_test_set], data.loc[in_test_set]

    if split_strategy == 'hash_ID':
        housing_with_id = housing.reset_index()  # adds an 'index' column
        train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')
    elif split_strategy == 'hash_ID_homemade':
        housing_with_id = housing
        housing_with_id['id'] = housing['longitude'] * 1000 + housing['latitude']
        train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'id')

elif split_strategy == 'stratified':
    # scale income (not strictly necessary)
    housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
    # split < 5 in 4 bins and fill rest in > 5 bin
    housing['income_cat'].where(housing['income_cat'] < 5, 5, inplace=True)

    print('Abundance in original dataset')
    print(housing['income_cat'].value_counts() / len(housing))

    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        train_set = housing.loc[train_index]
        test_set = housing.loc[test_index]

    print('Abundance in stratified test set')
    print(test_set['income_cat'].value_counts() / len(test_set))

    print('Abundance in stratified train set')
    print(train_set['income_cat'].value_counts() / len(train_set))

    for set_ in (train_set, test_set):
        set_.drop('income_cat', axis=1, inplace=True)

print(len(train_set), 'train +', len(test_set), 'test')

# -----------------------------------------------------------------------------
# DATA EXPLORATION
# -----------------------------------------------------------------------------

housing = train_set.copy()

if SAVE_PLOTS:
    print('Plot geographical information')
    plt.figure()
    housing.plot(kind='scatter', x='longitude', y='latitude')
    plt.savefig('simple_scatter.pdf')

    plt.figure()
    housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
                 s=housing['population']/100, label='population', figsize=(10, 7),
                 c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
    plt.legend()
    plt.savefig('beautyful_scatter.pdf')

print('Investigate correlations')

corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))

from pandas.plotting import scatter_matrix
attributes = ['median_house_value', 'median_income',
              'total_rooms', 'housing_median_age']

if SAVE_PLOTS:
    plt.figure()
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.savefig('scatter_matrix.png')

print('Experiment with attribute combinations')

housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']

corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))

# -----------------------------------------------------------------------------
# DATA PREPROCESSING
# -----------------------------------------------------------------------------

print('Preprocess data')

housing = train_set.drop('median_house_value', axis=1)
housing_labels = train_set['median_house_value'].copy()

print('Data cleaning')

housing_num = housing.drop('ocean_proximity', axis=1)

missing_strategy = 'imputer_fill_median'

if missing_strategy == 'drop_instances':
    housing = housing.dropna(subset=['total_bedrooms'])

elif missing_strategy == 'drop_columns':
    housing = housing.drop('total_bedrooms', axis=1)

elif missing_strategy == 'fill_median':
    median = housing['total_bedrooms'].median()
    housing['total_bedrooms'].fillna(median, inplace=True)

elif missing_strategy == 'imputer_fill_median':
    from sklearn.preprocessing import Imputer
    imputer = Imputer(strategy='median')
    imputer.fit(housing_num)
    print(imputer.statistics_)
    print(housing.median().values)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)

print('Convert text to numbers')

housing_cat = housing['ocean_proximity']
strategy = 'by_hand'

if strategy == 'by_hand':
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    housing_cat_encoded = encoder.fit_transform(housing_cat)
    print(housing_cat_encoded)
    print(encoder.classes_)

    from sklearn.preprocessing import OneHotEncoder
    hot_encoder = OneHotEncoder()
    housing_cat_1hot = hot_encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
    print(housing_cat_1hot)  # SciPy sparse matrix
    # housing_cat_1hot.toarray()  # (dense) NumPy array
elif strategy == 'automatically':
    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer(sparse_output=True)
    housing_cat_1hot = encoder.fit_transform(housing_cat)
    print(housing_cat_1hot)

print('Write transformer class for pipeline')

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        # no *args, **kargs to make use of BaseEstimator class
        # other args can be steered later as hyperparameters
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

print('Custom transformer dataframe -> NumPy array')

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names].values

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']


print('Custom label binarizer to make new version of LabelBinarizer work')

class LabelBinarizer_new(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.encoder = None
    def fit(self, X, y=0):
        return self
    def transform(self, X, y=0):
        if(self.encoder is None):
            print("Initializing encoder")
            self.encoder = LabelBinarizer()
            result = self.encoder.fit_transform(X)
        else:
            result = self.encoder.transform(X)
        return result

print('Pipeline for transformation and feature scaling')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer

num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),
                         ('imputer', Imputer(strategy='median')),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scaler', StandardScaler()),
                         ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)),
                         ('label_binarizer', LabelBinarizer_new()),
                         ])

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[('num_pipeline', num_pipeline),
                                               ('cat_pipeline', cat_pipeline),
                                               ])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
print(housing_prepared.shape)

# -----------------------------------------------------------------------------
# MODEL SELECTION
# -----------------------------------------------------------------------------

print('Train a linear regression')
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print('Predictions:', lin_reg.predict(some_data_prepared))
print('Labels:', list(some_labels))

print('Assess accuracy of model')
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print('RMSE:', lin_rmse)

print('Train decision tree')
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print('RMSE:', tree_rmse)

print('Train support vector machine')
from sklearn.svm import SVR
svm_reg = SVR()
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
print('RMSE:', svm_rmse)

print('Train random forest regression')
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print('RMSE:', forest_rmse)

print('Store models and their parameters')
storage_strategy = 'sklearn'

if storage_strategy == 'sklearn':
    from sklearn.externals import joblib
    joblib.dump(forest_reg, 'forest_reg_sklearn.pkl')
    # forest_reg_loaded = joblib.load('forest_reg_sklearn.pkl')
elif storage_strategy == 'pickle':
    import pickle
    pickle.dump(forest_reg, open('forest_reg_pickle.pkl', 'wb'))
    # forest_reg_loaded = pickle.load( open( "forest_reg_pickle.pkl", "rb" ) )

# -----------------------------------------------------------------------------
# CROSS VALIDATION
# -----------------------------------------------------------------------------

def display_scores(scores, label=''):
    if label:
        print('Scores for', label)
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())

from sklearn.model_selection import cross_val_score

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores, 'linear regression')

tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                              scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores(tree_rmse_scores, 'single decision tree')

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores, 'decision tree forest')

# -----------------------------------------------------------------------------
# MODEL TUNING
# -----------------------------------------------------------------------------

print('Tune the random forest model hyperparameters')

grid_strategy = 'grid_search'

if grid_strategy == 'grid_search':
    from sklearn.model_selection import GridSearchCV as SearchCV
elif grid_strategy == 'random_search':  # TODO: needs to be debugged
    from sklearn.model_selection import RandomizedSearchCV as SearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

forest_reg = RandomForestRegressor()
grid_search = SearchCV(forest_reg, param_grid, cv=5,
                       scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
extra_attribs = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances, attributes), reverse=True))

# -----------------------------------------------------------------------------
# EVALUATE ON TEST SET
# -----------------------------------------------------------------------------
print('Evaluate test set')
final_model = grid_search.best_estimator_

X_test = test_set.drop('median_house_value',axis=1)
y_test = test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print('RMSE on test set:', final_rmse)
