import os
import urllib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR

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

# Visualize the data
country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
plt.savefig('life_satisfaction.pdf')

# Select a linear model
regression_model = 'LinearRegression'
if regression_model == 'LinearRegression':
    lin_reg = LinearRegression()
elif regression_model == 'SVM':
    lin_reg = LinearSVR(epsilon=1.5)

# Train the model
print('Train the model')
lin_reg.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
result = lin_reg.predict(X_new)
print('Prediction for Cyprus:', result)