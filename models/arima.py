import statsmodels.api as sm
import pandas as pd

df = pd.read_csv('../data/processed/calendar_processed-May-13-2022.csv')
df = df["total_sales"]

print("Total nulls:", df.isnull().sum())
print("Dropping nulls...")
df = df.dropna()
model = sm.tsa.arima.ARIMA(df, order=(2, 1, 0))
results_ARIMA = model.fit()
print(results_ARIMA.summary())
results_ARIMA.save('results/model.pkl')
print("Model saved to model.pkl")
print("\n\n")

# Load model
from statsmodels.tsa.arima.model import ARIMAResults
model = ARIMAResults.load('model.pkl')
print(model.summary())
