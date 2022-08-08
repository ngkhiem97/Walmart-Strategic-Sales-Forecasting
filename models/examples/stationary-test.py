from statsmodels.tsa.stattools import adfuller
import pandas as pd

df = pd.read_csv('../../data/processed/calendar_processed-May-13-2022.csv')

def ad_test(dataset):
    print("Total nulls:", dataset.isnull().sum())
    print("Dropping nulls...")
    dataset = dataset.dropna()
    dftest = adfuller(dataset, autolag = 'AIC')
    print("1. ADF : ",dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression:",      dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t",key, ": ", val)

    # Write results to file
    with open('results/stationary-test-results.txt', 'w') as f:
        f.write("1. ADF : " + str(dftest[0]) + "\n")
        f.write("2. P-Value : " + str(dftest[1]) + "\n")
        f.write("3. Num Of Lags : " + str(dftest[2]) + "\n")
        f.write("4. Num Of Observations Used For ADF Regression: " + str(dftest[3]) + "\n")
        f.write("5. Critical Values :\n")
        for key, val in dftest[4].items():
            f.write("\t" + key + ": " + str(val) + "\n")
        
total_sales = df['total_sales']
ad_test(total_sales)

