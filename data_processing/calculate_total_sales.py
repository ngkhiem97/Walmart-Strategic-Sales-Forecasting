import pandas as pd
import time
from datetime import date

calendar_data = pd.read_csv('../data/raw/walmart_sales_data/calendar.csv')
validation_data = pd.read_csv('../data/raw/walmart_sales_data/sales_train_validation.csv')

start_time = time.time()
for index, row in calendar_data.iterrows():
    if index%100 == 0:
        print(f"{index} --- {time.time() - start_time} seconds ---")
    if row["d"] not in validation_data:
        continue
    calendar_data.loc[index, "total_sales_CA_HOBBIES"] = validation_data[(validation_data["cat_id"] == "HOBBIES") & (validation_data["state_id"] == "CA")][row["d"]].sum()
    calendar_data.loc[index, "total_sales_TX_HOBBIES"] = validation_data[(validation_data["cat_id"] == "HOBBIES") & (validation_data["state_id"] == "TX")][row["d"]].sum()
    calendar_data.loc[index, "total_sales_WI_HOBBIES"] = validation_data[(validation_data["cat_id"] == "HOBBIES") & (validation_data["state_id"] == "WI")][row["d"]].sum()
    calendar_data.loc[index, "total_sales_CA_HOUSEHOLD"] = validation_data[(validation_data["cat_id"] == "HOUSEHOLD") & (validation_data["state_id"] == "CA")][row["d"]].sum()
    calendar_data.loc[index, "total_sales_TX_HOUSEHOLD"] = validation_data[(validation_data["cat_id"] == "HOUSEHOLD") & (validation_data["state_id"] == "TX")][row["d"]].sum()
    calendar_data.loc[index, "total_sales_WI_HOUSEHOLD"] = validation_data[(validation_data["cat_id"] == "HOUSEHOLD") & (validation_data["state_id"] == "WI")][row["d"]].sum()
    calendar_data.loc[index, "total_sales_CA_FOODS"] = validation_data[(validation_data["cat_id"] == "FOODS") & (validation_data["state_id"] == "TX")][row["d"]].sum()
    calendar_data.loc[index, "total_sales_TX_FOODS"] = validation_data[(validation_data["cat_id"] == "FOODS") & (validation_data["state_id"] == "TX")][row["d"]].sum()
    calendar_data.loc[index, "total_sales_WI_FOODS"] = validation_data[(validation_data["cat_id"] == "FOODS") & (validation_data["state_id"] == "WI")][row["d"]].sum()
    calendar_data.loc[index, "total_sales"] = validation_data[row["d"]].sum()
print(f"1950 --- {time.time() - start_time} seconds ---")

today = date.today()
calendar_data.to_csv(f'../data/processed/calendar_processed-{today.strftime("%b-%d-%Y")}.csv')