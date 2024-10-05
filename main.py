import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Constants
DATA_PATH = '/Users/caioteixeira/PycharmProjects/etfperformance/portfolio_data/'
PORTFOLIO_PATH = DATA_PATH + 'path_to_your_yahoo_finance_portfolio.csv'
OUTPUT_PATH = DATA_PATH + 'predictions.csv'
YEARS_TO_PREDICT = [1, 5]

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

    for _, row in portfolio.iterrows():
        ticker = row['Symbol']
        etf = yf.Ticker(ticker)

        # Fetch historical data
        data = etf.history(period='ytd')
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
                # Fetch the exchange rate using yfinance for EUR/USD (or equivalent for other currencies)
                # Here we assume we're converting to USD from EUR as an example
                fx_ticker = f"{currency}USD=X"
                fx_exchange = yf.Ticker(fx_ticker)
                fx_data = fx_exchange.history(period='1d')
                exchange_rate = fx_data['Close'].iloc[-1]
                last_close = last_close * exchange_rate
                ma50 = ma50 * exchange_rate


            except Exception as e:
                print(f"Could not fetch exchange rate for {currency} to USD: {e}")
                # Keep the original price if conversion fails

        # Update the portfolio DataFrame
        portfolio.loc[portfolio['Symbol'] == ticker, 'Last_Close'] = last_close
        portfolio.loc[portfolio['Symbol'] == ticker, 'Current Price'] = last_close
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
    series.index = pd.to_datetime(series.index)
    # Add freq
    series = series.asfreq('D')
    series = series.fillna(method='ffill')
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=False)
    forecast = results.forecast(steps=steps)
    return forecast


def predict_portfolio(portfolio, etf_data, years):
    """
    Predict future values for each ETF in the portfolio, converting to USD if necessary.

    Args:
    portfolio (pd.DataFrame): DataFrame containing portfolio details with a 'Symbol' column.
    etf_data (dict): Dictionary where keys are ETF symbols and values are price series.
    years (int): Number of years to predict into the future.

    Returns:
    list: A list of dictionaries containing predictions for each ETF, converted to USD if needed.
    """
    predictions = []
    steps = int(years * 365)  # Assuming daily predictions

    for index, row in portfolio.iterrows():
        ticker = row['Symbol']
        if ticker in etf_data:
            series = etf_data[ticker]
            forecast = predict_sarima(series, steps)

            # Obter informações do ETF diretamente dentro da função
            try:
                etf = yf.Ticker(ticker)
                etf_info = etf.info
                currency = etf_info.get('currency', 'USD')
            except Exception as e:
                print(f"Failed to retrieve info for {ticker}: {e}")
                currency = 'USD'  # Default to USD if info can't be retrieved

            # Preparar o dicionário de predições
            pred_dict = {
                'Index': index,
                'Symbol': ticker,
                'today': series.iloc[-1],  # Último valor da série
                **{f"{y} years": forecast.iloc[int(y * 365) - 1] for y in YEARS_TO_PREDICT}
            }

            #print(f"Currency for {ticker}: {currency}")

            # Converter o preço para USD, se necessário
            if currency != 'USD':
                try:
                    fx_ticker = f"{currency}USD=X"
                    fx_exchange = yf.Ticker(fx_ticker)
                    fx_data = fx_exchange.history(period='1d')
                    exchange_rate = fx_data['Close'].iloc[-1]
                    pred_dict['today'] *= exchange_rate
                    for key in [k for k in pred_dict.keys() if 'years' in k]:
                        pred_dict[key] *= exchange_rate
                except Exception as e:
                    print(f"Failed to convert {ticker} from {currency} to USD: {e}. Using original currency.")

            predictions.append(pred_dict)
        else:
            print(f"No data found for ticker: {ticker}")

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


def visualize_statistics(df_predictions):
    """
    Visualize quantity * price for today, 5, 10, and 15 years for each ETF.

    This function creates:
    1. A table showing ETF portfolio values over time.
    2. A pie chart for ETF breakdown based on today's value.
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

    # Create table
    table_data = grouped[['Symbol', 'Total Today'] + [f'Total {year} years' for year in YEARS_TO_PREDICT]]

    # Calculate totals for each column
    total_row = table_data.sum().to_frame().T
    total_row['Symbol'] = 'Total'

    # Concatenate total row to the DataFrame
    table_data = pd.concat([table_data, total_row], ignore_index=True)

    # Format numbers with dollar sign and no conversion to billions
    for col in table_data.columns[1:]:
        table_data[col] = table_data[col].apply(lambda x: f'${x:.2f}')

    # Print table
    print(table_data.to_string(index=False))

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
    visualize_statistics(df_predictions)