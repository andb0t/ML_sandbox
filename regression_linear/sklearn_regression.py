import os
import sys
import urllib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVR

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utils'))
import my_plots


GDP_URL = 'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/lifesat/gdp_per_capita.csv'
SAT_URL = 'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/lifesat/oecd_bli_2015.csv'


def download_from_URL(from_URL, to_URL=None):
    if not to_URL:
        to_URL = from_URL.split('/')[-1]
    if os.path.isfile(to_URL):
        print('File already present, skip download!')
        return
    print('Downloading from', from_URL)
    urllib.request.urlretrieve(from_URL, to_URL)

# Download data
download_from_URL(GDP_URL)
download_from_URL(SAT_URL)

# Load the data
oecd_bli = pd.read_csv('oecd_bli_2015.csv', thousands=',')
gdp_per_capita = pd.read_csv('gdp_per_capita.csv', thousands=',', delimiter='\t',
                             encoding='latin1', na_values='n/a')

# Verify datasets
print('Peek into oecd_bli:')
print(oecd_bli.head())
print('Peek into gdp_per_capita:')
print(gdp_per_capita.head())

# Prepare the data


def prepare_country_stats(sat_data, gdp_data):
    sat_data = sat_data[sat_data["INEQUALITY"] == "TOT"]
    sat_data = sat_data.pivot(index="Country", columns="Indicator", values="Value")
    gdp_data.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_data.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=sat_data, right=gdp_data,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats['GDP per capita']]
y = np.c_[country_stats['Life satisfaction']]

print('Scale X')
X = X / 10000

# Visualize the data
country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
plt.savefig('life_satisfaction.pdf')

# Select a linear model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

print(type(X))
print(type(X[0]))
print(X)
print(y)

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
result = lin_reg.predict(X_new)
print('Prediction for Cyprus:', result)

svm_reg = LinearSVR(epsilon=0.5)  # needs feature scaling
sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
ridge_reg = Ridge(alpha=1, solver='cholesky')
lasso_reg = Lasso(alpha=0.1)
elnet_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)

print('Train all of them and check their accuracy')
# headers = ['Algorithm', 'Accuracy']
# table = []
for reg in (lin_reg, svm_reg, sgd_reg, ridge_reg, lasso_reg, elnet_reg):
    name = reg.__class__.__name__
    print('Fitting', name)
    reg.fit(X, y)
    # y_pred = reg.predict(X)
    # accuracy = round(accuracy_score(y_test, y_pred), 3)
    # table.append([name, accuracy])
    my_plots.plot_reg_train_scatter(
        X, y, reg,
        title=name,  # + ' (' + str(accuracy) + ')',
        save_name='reg_model_' + name + '.png')

# print(tabulate.tabulate(sorted(table, key=lambda tup: tup[1], reverse=True),
#                         headers=headers, tablefmt='grid'))
