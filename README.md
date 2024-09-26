# ETF Portfolio Prediction Project

This project aims to predict the future performance of ETFs (Exchange Traded Funds) based on historical data, using the SARIMA (Seasonal Autoregressive Integrated Moving Average) model. The script processes an ETF portfolio, fetches historical data from Yahoo Finance, predicts future values for up to 15 years, and generates statistics for the portfolio's predicted performance.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Workflow](#workflow)
   1. [Step 1: Read and Normalize Portfolio](#step-1-read-and-normalize-portfolio)
   2. [Step 2: Fetch and Enrich ETF Data](#step-2-fetch-and-enrich-etf-data)
   3. [Step 3: Save Enriched Portfolio](#step-3-save-enriched-portfolio)
   4. [Step 4: Predict ETF Prices with SARIMA](#step-4-predict-etf-prices-with-sarima)
   5. [Step 5: Save Predictions](#step-5-save-predictions)
   6. [Step 6: Print Portfolio Statistics](#step-6-print-portfolio-statistics)
4. [SARIMA Model Explanation](#sarima-model-explanation)
5. [File Structure](#file-structure)
6. [Conclusion](#conclusion)

## Overview

This Python script automates the analysis of an ETF portfolio and provides predictions of future values. The project follows these key steps:

- Reading and normalizing the portfolio data.
- Fetching ETF historical data from Yahoo Finance.
- Enriching the portfolio with calculated metrics (e.g., Moving Average, Volatility, etc.).
- Predicting the future performance of the ETFs using the SARIMA model.
- Saving the predictions to a CSV file.
- Printing summarized statistics for the portfolio today and in the future.

## Requirements

Before running the script, make sure you have the following dependencies installed:

- `yfinance`: To fetch historical stock and ETF data.
- `pandas`: For data manipulation.
- `numpy`: For numerical computations.
- `statsmodels`: To use the SARIMA model for time series prediction.

You can install these libraries via pip:

```bash
pip install yfinance pandas numpy statsmodels
```

## Workflow

### Step 1: Read and Normalize Portfolio

Function: `read_and_normalize_portfolio(file_path)`

The first step is to load the portfolio data from a CSV file. The script expects the portfolio to have a column named `Weight`, which will be normalized to ensure that the total weights sum to 1. 

```python
portfolio = read_and_normalize_portfolio(PORTFOLIO_PATH)
```

#### Input:
- CSV file containing ETF symbols, weights, and other necessary information.

#### Output:
- Normalized portfolio DataFrame.

### Step 2: Fetch and Enrich ETF Data

Function: `fetch_and_enrich_etf_data(portfolio)`

This step fetches historical data for each ETF from Yahoo Finance using the `yfinance` API. For each ETF, the following metrics are calculated and added to the portfolio:

- **Last Close Price**: The latest closing price of the ETF.
- **50-Day Moving Average (MA50)**: Average of the last 50 days of prices.
- **Volatility**: Calculated using the standard deviation of the daily returns.
- **Expense Ratio**: The ETF's expense ratio, if available.
- **Currency Conversion**: Prices are converted to USD if necessary using an exchange rate API (if the ETF is priced in another currency).

```python
portfolio, etf_data = fetch_and_enrich_etf_data(portfolio)
```

#### Output:
- Enriched portfolio DataFrame with additional columns for the calculated metrics.
- A dictionary of historical closing prices for each ETF.

### Step 3: Save Enriched Portfolio

Function: `save_enriched_portfolio(portfolio)`

After enriching the portfolio, the data is saved to a CSV file (`enriched_portfolio.csv`). This provides a reference for the next steps and allows for tracking the portfolio's data.

```python
save_enriched_portfolio(portfolio)
```

### Step 4: Predict ETF Prices with SARIMA

Function: `predict_portfolio(portfolio, etf_data, years)`

The core of the project is to predict future ETF prices. This is done using the SARIMA model, which is well-suited for time series forecasting. The prediction is made for 5, 10, and 15 years ahead (as defined by `YEARS_TO_PREDICT`).

SARIMA parameters used:
- `order=(1,1,1)`: Non-seasonal components (AR, differencing, MA).
- `seasonal_order=(1,1,1,12)`: Seasonal components (AR, differencing, MA, seasonal period).

The model predicts daily prices for the number of years specified and extracts values for the specified future years.

```python
portfolio_predictions = predict_portfolio(portfolio, etf_data, max(YEARS_TO_PREDICT))
```

#### Output:
- A list of predictions for each ETF in the portfolio.

### Step 5: Save Predictions

Function: `save_predictions(portfolio, predictions)`

The predictions generated in the previous step are saved to a CSV file (`predictions.csv`). This file contains both the current ETF data and the predicted values for future years.

```python
save_predictions(portfolio, portfolio_predictions)
```

### Step 6: Print Portfolio Statistics

Function: `print_statistics(df_predictions)`

Finally, the script prints statistics for the portfolio. The total value of the portfolio is calculated for today and for the future years (5, 10, and 15 years). It also prints a detailed breakdown of each ETF's performance.

```python
df_predictions = pd.read_csv(OUTPUT_PATH, index_col=0)
print_statistics(df_predictions)
```

#### Output:
- Console output of the portfolio's total value today and the predicted future values.
- Breakdown of each ETF's quantity and total predicted value.

## SARIMA Model Explanation

The SARIMA model used for prediction is a powerful time series forecasting method that takes into account both seasonal and non-seasonal factors. The model is defined by two sets of parameters:

- **Non-seasonal components (`p`, `d`, `q`)**:
  - `p`: Number of lag observations included (AutoRegressive part).
  - `d`: Degree of differencing (number of times the data is differenced).
  - `q`: Size of the moving average window.
  
- **Seasonal components (`P`, `D`, `Q`, `m`)**:
  - `P`: Seasonal autoregressive terms.
  - `D`: Seasonal differencing.
  - `Q`: Seasonal moving average terms.
  - `m`: Number of time steps for a single seasonal period (12 for monthly data).

## File Structure

```
/Users/caioteixeira/PycharmProjects/etfperformance/portfolio_data/
│
├── path_to_your_yahoo_finance_portfolio.csv  # Input: Initial ETF portfolio
├── enriched_portfolio.csv                    # Output: Portfolio with enriched ETF data
├── predictions.csv                           # Output: Portfolio with future predictions
└── main.py                                   # Python script (this project)
```

## Conclusion

This project provides a framework for predicting the future performance of an ETF portfolio using historical data and time series forecasting. By leveraging the SARIMA model and Yahoo Finance data, you can gain insights into the potential future value of your portfolio.

