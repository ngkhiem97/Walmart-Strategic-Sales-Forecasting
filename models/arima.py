from statsmodels.tsa.arima_model import ARIMA

# Doing ARIMA modeling
model = ARIMA(data, order=(2, 1, 0))
results_ARIMA = model.fit(disp=False)