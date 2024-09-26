import pandas as pd
from datetime import datetime
import sqlite3


def clean_and_analyze_yahoo_portfolio_csv(file_path):
    """
    Read, clean, and analyze Yahoo Finance portfolio CSV for gains/losses calculation.

    Args:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Cleaned DataFrame with gains/loss calculations per ticker.
    """
    # Read the CSV
    df = pd.read_csv(file_path)

    # Rename columns for clarity
    df.columns = ['ticker', 'current_price', 'date', 'time', 'change', 'open', 'high', 'low',
                  'volume', 'trade_date', 'purchase_price', 'quantity', 'commission',
                  'high_limit', 'low_limit', 'comment']

    # Convert necessary columns to appropriate types
    df['date'] = pd.to_datetime(df['date'])
    df['current_price'] = pd.to_numeric(df['current_price'], errors='coerce')
    df['purchase_price'] = pd.to_numeric(df['purchase_price'], errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')

    # Drop rows with NAs in key columns
    df = df.dropna(subset=['current_price', 'purchase_price', 'quantity'])

    # Calculate total purchase value and current value for each entry
    df['total_purchase_value'] = df['purchase_price'] * df['quantity']
    df['total_current_value'] = df['current_price'] * df['quantity']

    # Group by ticker to aggregate values
    grouped = df.groupby('ticker').agg({
        'total_purchase_value': 'sum',
        'total_current_value': 'sum',
        'quantity': 'sum'
    }).reset_index()

    # Calculate gain/loss
    grouped['gain_loss'] = grouped['total_current_value'] - grouped['total_purchase_value']
    grouped['percent_gain'] = (grouped['gain_loss'] / grouped['total_purchase_value']) * 100

    return grouped


# Usage
if __name__ == "__main__":
    file_path = '../portfolio_data/path_to_your_yahoo_finance_portfolio.csv'
    result_df = clean_and_analyze_yahoo_portfolio_csv(file_path)

    print(result_df)
    print(f"Total Portfolio Value: ${result_df['total_current_value'].sum():.2f}")
    print(f"Total Gain/Loss: ${result_df['gain_loss'].sum():.2f}")

    # Save to SQLite
    conn = sqlite3.connect('portfolio.db')
    result_df.to_sql('portfolio', conn, if_exists='append', index=False)
    conn.close()

# Usage
if __name__ == "__main__":
    file_path = '../portfolio_data/path_to_your_yahoo_finance_portfolio.csv'
    result_df = clean_and_analyze_yahoo_portfolio_csv(file_path)

    print(result_df)
    print(f"Total Portfolio Value: ${result_df['total_current_value'].sum():.2f}")
    print(f"Total Gain/Loss: ${result_df['gain_loss'].sum():.2f}")

    # Save to SQLite
    conn = sqlite3.connect('portfolio.db')
    result_df.to_sql('portfolio', conn, if_exists='replace', index=False)

    # Export tickers for use in the second script
    tickers = result_df['ticker'].tolist()
    with open('../portfolio_data/portfolio_tickers.txt', 'w') as f:
        for ticker in tickers:
            f.write(f"{ticker}\n")

    conn.close()