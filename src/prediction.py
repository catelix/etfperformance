import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import os
from datetime import datetime


def get_latest_csv(csv_dir):
    """
    Find the most recent CSV file in the directory.

    :param csv_dir: Directory path to search for CSV files
    :return: Path to the most recent CSV file
    """
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the specified directory")

    latest_file = max(csv_files, key=lambda f: os.path.getmtime(os.path.join(csv_dir, f)))
    return os.path.join(csv_dir, latest_file)


def read_from_csv(csv_path):
    """
    Read ETF data from CSV file, ensuring timezone consistency.

    :param csv_path: Path to the CSV file
    :return: pd.DataFrame, ETF data with 'Date' as index and 'MS' frequency
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=[0])
    df.index.name = 'Date'

    # Convert to UTC if timezone info is present
    df.index = pd.to_datetime(df.index, utc=True)

    # Ensure monthly frequency, filling gaps with last known value
    df = df.asfreq('MS').fillna(method='ffill')

    return df


def predict_future_prices(df, horizon=15):
    """
    Predict future ETF prices using SARIMA model.

    :param df: DataFrame with ETF data
    :param horizon: Number of years to predict into the future
    :return: DataFrame with predicted prices
    """
    future_index = pd.date_range(start=df.index[-1], periods=horizon * 12 + 1, freq='MS')
    predicted_prices = pd.DataFrame(index=future_index)

    for ticker in df.columns:
        if not ticker.endswith(('_Return', '_Volatility', '_MA50')):
            model = SARIMAX(df[ticker], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=horizon * 12)
            predicted_prices[ticker] = forecast

    return predicted_prices


if __name__ == "__main__":
    csv_dir = '/portfolio_data'
    csv_path = get_latest_csv(csv_dir)

    try:
        df = read_from_csv(csv_path)
        print("Available columns:", df.columns.tolist())

        future_prices = predict_future_prices(df)

        # Save predictions to a new CSV file with a timestamp
        output_csv_path = os.path.join(csv_dir, f'predicted_prices_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        future_prices.to_csv(output_csv_path)

        print(f"Predicted prices saved to {output_csv_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure CSV files exist in the specified directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")