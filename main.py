import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
import os

# Constants
DATA_PATH = '/Users/caioteixeira/PycharmProjects/etfperformance/portfolio_data/'
PORTFOLIO_PATH = DATA_PATH + 'path_to_your_yahoo_finance_portfolio.csv'
OUTPUT_PATH = DATA_PATH + 'predictions.csv'
YEARS_TO_PREDICT = [5, 10, 15]


def read_and_normalize_portfolio(file_path):
    """
    Read the ETF portfolio from CSV and normalize data if needed.
    """
    portfolio = pd.read_csv(file_path)

    # Example of normalization: Here we just assume the 'Weight' column exists and sum to 1
    if 'Weight' in portfolio.columns:
        portfolio['Weight'] = portfolio['Weight'] / portfolio['Weight'].sum()

    return portfolio


def fetch_and_enrich_etf_data(portfolio):
    """
    Fetch ETF historical data from Yahoo Finance and enrich the portfolio DataFrame
    with Last_Close, MA50, Volatility, Expense Ratio, and convert prices to USD if necessary.
    """
    etf_data = {}
    base_url = "https://api.exchangeratesapi.io/latest?base="

    for _, row in portfolio.iterrows():
        ticker = row['Symbol']
        etf = yf.Ticker(ticker)

        # Fetch historical data
        data = etf.history(period='5y')
        #data = etf.history(period='max')
        etf_data[ticker] = data['Close']

        # Get the currency of the ETF
        currency = etf.info.get('currency', 'USD')  # Default to USD if not available

        # Calculate required metrics
        last_close = data['Close'].iloc[-1]
        ma50 = data['Close'].rolling(window=50).mean().iloc[-1]
        volatility = data['Close'].pct_change().dropna().std() * np.sqrt(252)

        # Get the expense ratio
        expense_ratio = etf.info.get('expenseRatio', 0)

        # Convert price to USD if necessary
        if currency != 'USD':
            try:
                response = requests.get(f"{base_url}{currency}")
                response.raise_for_status()  # Will raise an HTTPError if the status is 4xx, 5xx
                rates = response.json()['rates']
                exchange_rate = rates['USD']
                last_close = last_close * exchange_rate
            except requests.exceptions.RequestException as e:
                print(f"Could not fetch exchange rate for {currency}: {e}")
                # Keep the original price if conversion fails

        # Update the portfolio DataFrame
        portfolio.loc[portfolio['Symbol'] == ticker, 'Last_Close'] = last_close
        portfolio.loc[portfolio['Symbol'] == ticker, 'MA50'] = ma50
        portfolio.loc[portfolio['Symbol'] == ticker, 'Volatility'] = volatility
        portfolio.loc[portfolio['Symbol'] == ticker, 'Expense_Ratio'] = expense_ratio
        portfolio.loc[portfolio['Symbol'] == ticker, 'Currency'] = currency  # Add currency column

    return portfolio, etf_data

def save_enriched_portfolio(portfolio):
    """
    Save the portfolio enriched with latest ETF data to a CSV file.
    """
    enriched_path = DATA_PATH + 'enriched_portfolio.csv'
    portfolio.to_csv(enriched_path, index=False)
    print(f"Enriched portfolio saved to {enriched_path}")


def predict_sarima(series, steps):
    """
    Fit SARIMA model and predict future values.
    """
    # Note: This is a basic setup. In practice, you might need to select optimal parameters.
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=False)
    forecast = results.forecast(steps=steps)
    return forecast


def predict_portfolio(portfolio, etf_data, years):
    """
    Predict future values for each ETF in the portfolio. Handles repeated tickers.
    """
    predictions = []
    for index, row in portfolio.iterrows():
        ticker = row['Symbol']
        steps = int(years * 365)  # Assuming daily predictions
        if ticker in etf_data:
            series = etf_data[ticker]
            forecast = predict_sarima(series, steps)
            predictions.append({
                'Index': index,
                'Symbol': ticker,
                'today': series.iloc[-1],  # Latest value
                **{f"{y} years": forecast[int(y * 365) - 1] for y in YEARS_TO_PREDICT}
            })
    return predictions

def save_predictions(portfolio, predictions):
    """
    Save ETF predictions along with portfolio data to a CSV file.
    Each portfolio row (even with repeated tickers) is saved.
    """
    df_predictions = pd.DataFrame(predictions)

    # Merge the predictions with the original portfolio data
    portfolio_with_predictions = portfolio.merge(
        df_predictions[['Index', 'today', *[f"{y} years" for y in YEARS_TO_PREDICT]]],
        left_index=True,
        right_on='Index',
        how='left'
    ).drop(columns=['Index'])  # Drop 'Index' column as it's no longer needed

    # Save to CSV
    portfolio_with_predictions.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_PATH}")


def print_statistics(df_predictions):
    """
    Calculate quantity * price for today, 5, 10, and 15 years for each row.
    Then, print the results grouped by ETF.
    """
    # Calculate total value for today and future periods for each row
    df_predictions['Total Today'] = df_predictions['Quantity'] * df_predictions['today']
    for year in YEARS_TO_PREDICT:
        df_predictions[f'Total {year} years'] = df_predictions['Quantity'] * df_predictions[f'{year} years']

    # Group by ETF (Symbol) and sum the quantities and total values
    grouped = df_predictions.groupby('Symbol').agg({
        'Quantity': 'sum',
        'Total Today': 'sum',
        **{f'Total {year} years': 'sum' for year in YEARS_TO_PREDICT}
    }).reset_index()

    # Print total portfolio statistics
    total_today = grouped['Total Today'].sum()
    print(f"Total Portfolio Value Today: ${total_today:.2f}")

    for year in YEARS_TO_PREDICT:
        total_future = grouped[f'Total {year} years'].sum()
        print(f"Predicted Total Portfolio Value in {year} years: ${total_future:.2f}")

    # Print ETF breakdown
    print("\n--- ETF Breakdown ---")
    for _, row in grouped.iterrows():
        print(f"\nETF: {row['Symbol']}")
        print(f"  Quantity: {row['Quantity']}")
        print(f"  Total Value Today: ${row['Total Today']:.2f}")
        for year in YEARS_TO_PREDICT:
            print(f"  Predicted Total Value in {year} years: ${row[f'Total {year} years']:.2f}")

if __name__ == "__main__":
    # Step 1: Read and normalize portfolio
    portfolio = read_and_normalize_portfolio(PORTFOLIO_PATH)

    # Step 2: Fetch and enrich data
    portfolio, etf_data = fetch_and_enrich_etf_data(portfolio)

    # Step 3: Save enriched portfolio
    save_enriched_portfolio(portfolio)

    # Step 4: Run prediction model for each ETF
    portfolio_predictions = predict_portfolio(portfolio, etf_data, max(YEARS_TO_PREDICT))

    # Step 5: Save predictions
    save_predictions(portfolio, portfolio_predictions)

    # Step 6: Print statistics
    df_predictions = pd.read_csv(OUTPUT_PATH, index_col=0)
    print_statistics(df_predictions)