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

class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.5 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]

tscv = BlockingTimeSeriesSplit(n_splits=5)

print("Data length: " + str(len(X)))

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    print(str(X_train.shape[0]) + ": " + str(X_test.shape[0]))
    print("----------")