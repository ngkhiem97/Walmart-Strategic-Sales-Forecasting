import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.feature_selection import r_regression, f_regression
import matplotlib.pyplot as plt

class LinearRegressionForecast:
    def __init__(self, test_size=30):
        self.test_size = test_size

    def get_features_and_target(self, df, onehotencoding, target_feature='store_sales'):
        # create dummy features
        if onehotencoding:
            df = (
                df
                .join([pd.get_dummies(df[col], prefix=col) for col in onehotencoding])
                .drop(onehotencoding, axis=1)
                .rename(str.lower, axis=1)
            )

        # split the independent features and the target into X and y

        X = df.drop([target_feature], axis=1)
        y = df.loc[:, target_feature]

        return X, y

    def split_train_test(self, X, y):
        # create train-test set split.
        X_train = X[:-self.test_size]
        X_test = X[-self.test_size:]
        y_train = y[:-self.test_size]
        y_test = y[-self.test_size:]
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train, model, scaler=StandardScaler()):
        """ Trains model

        X: pandas DataFrame of the features
        y: pandas Series of the target variable
        model: sklearn model
        """

        # train the model.
        pipeline = Pipeline(steps = [
            ('imputer', SimpleImputer()),
            ('scaler', scaler),
            ('model', model)
        ])
        pipeline = pipeline.fit(X_train, y_train)
        return pipeline, pipeline.named_steps['model']

    def evaluate(self, pipeline, X_train, X_test, y_train, y_test, plot=True):
        # evaluate the model.
        y_train_pred = pipeline.predict(X_train)
        y_pred = pipeline.predict(X_test)
        score_train = r2_score(y_train, y_train_pred)
        score = r2_score(y_test, y_pred)
        print(f'Train score: {score_train.round(2)}')
        print(f'Test score: {score.round(2)}\n')
        if plot:
            self.plot_pred_vs_test(y_pred, y_test)
        return score

    def get_feature_correlation(self, X, y):
        X = StandardScaler().fit_transform(X, y)
        return r_regression(X, y)
    
    def get_feature_correlation_f(self, X, y):
        X = StandardScaler().fit_transform(X, y)
        return f_regression(X, y)

    def split_train_and_plot(self, df, cat_cols, model=LinearRegression(),):
        # split X and y
        X, y = self.get_features_and_target(df=df, onehotencoding=cat_cols)
        X_train, X_test, y_train, y_test = self.split_train_test(X, y)
        # train model
        pipeline, trained_model = self.train_model(X_train, y_train, model)
        score = self.evaluate(pipeline, X_train, X_test, y_train, y_test)
        return X, y, trained_model, score

    def plot_pred_vs_test(self, y_pred, y_test):
        y = pd.DataFrame(y_test)
        y['pred'] = y_pred
        y.plot(figsize=(8, 6))

lr_forecast = LinearRegressionForecast()
