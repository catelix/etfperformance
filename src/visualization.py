import pandas as pd
from datetime import datetime, timedelta


def read_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Columns in the DataFrame: {df.columns}")

    if 'Symbol' not in df.columns:
        print("Error: 'Symbol' column not found. Please check the CSV file for the correct column name.")
        return None

    # Identify and process date columns
    date_columns = [col for col in df.columns if '+00:00' in col]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df = df.dropna(subset=[col])  # Remove rows with failed date conversions

    print(f"DataFrame shape after date conversion: {df.shape}")

    # Melt DataFrame to restructure date and price columns
    df_melted = pd.melt(df, id_vars=['Symbol'],
                        value_vars=date_columns,
                        var_name='Date', value_name='Predicted Price')
    df_melted['Date'] = pd.to_datetime(df_melted['Date'])
    df_melted = df_melted.sort_values('Date')

    print(f"Melted DataFrame shape: {df_melted.shape}")
    return df_melted


def calculate_future_prices(df):
    current_date = datetime.now()
    results = []

    for symbol, group in df.groupby('Symbol'):
        if group.empty:
            print(f"No data for symbol: {symbol}")
            continue

        latest_price = group.loc[group['Date'].idxmax(), 'Predicted Price']

        for year in [0, 5, 10]:
            future_date = current_date + timedelta(days=year * 365)
            price = group.loc[group['Date'] <= future_date, 'Predicted Price'].iloc[-1] if not group.empty else None
            results.append({
                'Symbol': symbol,
                'Years': year,
                'Price': price,
                'Date': future_date.date() if year > 0 else current_date.date()
            })

    return pd.DataFrame(results).pivot(index='Symbol', columns='Years', values='Price').reset_index()


def print_results(df):
    for _, row in df.iterrows():
        print(f"Symbol: {row['Symbol']}, Current Price: {row[0]}, 5Y Price: {row[5]}, 10Y Price: {row[10]}")


def main(file_path):
    df = read_and_clean_data(file_path)
    if df is not None:
        future_prices = calculate_future_prices(df)
        print_results(future_prices)


if __name__ == "__main__":
    file_path = '/portfolio_data/new_portfolio_with_predictions_20240925_154155.csv'  # Update this path
    main(file_path)