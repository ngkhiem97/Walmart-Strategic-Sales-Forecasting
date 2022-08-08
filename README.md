# Walmart Strategic Sales Forecasting

This github repository holds the details for our university capstone project Walmart Strategic Sales Forecasting at Drexel University.

## Background

In any supply chain, especially retail, forecasting is essential for all parties in the business. For the business owner, it gives them the ability to make informed business decisions and develop data-driven strategies. By using current and historical data, businesses can predict future trends and forecasts and using those insights to plan resources, make appropriate adjustments to business strategy, lower the overall business operations cost, and increase profits. From the consumer standpoint, it helps reduce the overall costs for the business and get the right products that they want to the shelf. For the business' supply partners, this allows them to be proactive instead of reactive; they can produce the right amount ahead of time and reduce excesses.  

In this capstone project for the capstone project of the MS Data Science program at Drexel University, we used machine learning (ML) to solve a time-series forecasting problem of predicting sales at Walmart stores. Walmart is a retail business that gets products from its supply partners to deliver to consumers. We will use Walmart sales data from 3 states across the United States (California, Texas, and Wisconsin) over several years to predict future sales. 

## Contributors/Project Members
* Xi Chen: MS Data Science at Drexel University
* Emily Wang: MS Data Science at Drexel University
* Kriti Bartaria: MS Data Science at Drexel University
* Khiem Nguyen: MS Data Science at Drexel University

## Data sources

### 1. Walmart Sales Data:

We utilized an existing dataset present at [Kaggel.com](https://www.kaggle.com/code/konradb/ts-4-sales-and-demand-forecasting/data), it includes item-level details, department, product categories, and store details for stores in three US states (California, Texas, and Wisconsin). It also includes explanatory variables like price, promotions, day of the week, and special events. The dataset is arranged in a hierarchal order. 

### 2. Google Trends Data: 

[Google Trends](https://trends.google.com/trends/?geo=US) data is available for download via its web interface, but because we need to send multiple queries containing different keywords, geographical locations, and timeframes, we needed a more scalable approach. Therefore, we opted to use a Python package called Pytrends. Pytrends is an unofficial API (Application Programming Interface) for Google Trends. It allows us to automate the process of querying and downloading reports from Google Trends with Python scripts. With the API, we pulled data for interest over time with our desired timeframe, keywords, and geographic location. 

### 3. Unemployment Data

For this project, we have manually downloaded the seasonally adjusted unemployment data ranging daily from January 2011 to December 2016 from the [U.S Bureau of Labor & Statistics site](https://beta.bls.gov/dataViewer/view/timeseries/LNS14000000).  By seasonality it means that the periodic fluctuations associated with events such as weather, holidays, and the opening and closing of schools were also taken into consideration for the unemployment rate. 

### 4. Consumer Price Index (CPI)

The CPI data was also manually downloaded from the [U.S Bureau of Labor & Statistics site](https://www.bls.gov/cpi/data.htm) ranging from January 2011 to December 2016.

### 5. Gas Price Data 

[Gas Price Data](https://www.kaggle.com/datasets/mruanova/us-gasoline-and-diesel-retail-prices-19952021?select=PET_PRI_GND_DCUS_NUS_W.csv) is available for download from the Kaggle project website. It has a single csv file including the price of multiple gasoline types. 

## Predictive Models

* Linear Regression
* Long short-term memory (LSTM)
* Autoregressive Integrated Moving Average (ARIMA)

## Project files structure
* analysis: python notebooks of the Exploratory Data Analysis
* data-processing: python code to process the data
* data: raw and processed data
* models: machine learning model for sales prediction
* notebooks: additional python notebooks related to the project

## Resources
* [Cross Validation in Time Series](https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4)
* [Time Series Cross-Validation](https://goldinlocks.github.io/Time-Series-Cross-Validation/)
* [Building a Tractable, Feature Engineering Pipeline for Multivariate Time Series](https://www.kdnuggets.com/2022/03/building-tractable-feature-engineering-pipeline-multivariate-time-series.html)
* [Feature Engineering for Multivariate Stock Market Prediction with Python](https://www.relataly.com/feature-engineering-for-multivariate-time-series-models-with-python/1813/#h-feature-engineering-for-stock-market-forecasting-borrowing-features-from-chart-analysis)