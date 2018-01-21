import os
import tarfile
from six.moves import urllib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAVE_PLOTS = False

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

print('Split into train and test data set')

split_strategy = 'sklearn_naive'

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
