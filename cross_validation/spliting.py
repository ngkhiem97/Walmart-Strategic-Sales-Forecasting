import sys
sys.path.insert(0, '/home/khiem/Dropbox/Education/Drexel/DSCI-591/Project/Walmart-Strategic-Sales-Forecasting')

from sklearn.model_selection import TimeSeriesSplit
from data_processing.pre_modeling import pre_modeling
import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/CA_1_sales_data.csv', index_col=0, parse_dates=True)
df_processed = pre_modeling(df)

X = df_processed.drop('store_sales', axis=1).to_numpy()
Y = df_processed['store_sales'].to_numpy()

tscv = TimeSeriesSplit(test_size=30)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
