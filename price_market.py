# price_market.py
# ----------------------
# This module provides utility functions to:
# 1. Read hourly energy prices from an Excel file (e.g. spot market data)
# 2. Retrieve the price at a specific time index
#
# Inputs:
# - Excel file with energy prices (e.g. "EnergyPriceManyDays.xlsx")
#   Columns: [Date, Price]
#
# Outputs:
# - DataFrame with cleaned date and float-converted price
# - Single price value (in €/Wh) for a given time index

import pandas as pd

def read_energy_prices(filepath: str) -> pd.DataFrame:
    """
    Reads the Excel file and returns a DataFrame with 'date' and 'price' columns,
    limited to the first 480 rows (i.e., 20 days of hourly data).

    Parameters:
    - filepath (str): Path to the Excel file

    Returns:
    - pd.DataFrame: DataFrame with 'date' as datetime and 'price' as float [€/MWh]
    """
    # Load only first 480 rows and first two columns (date and price)
    df = pd.read_excel(filepath, usecols=[0, 1], nrows=480)

    # Rename columns for clarity
    df.columns = ['date', 'price']

    # Convert date column to datetime objects
    df['date'] = pd.to_datetime(df['date'])

    # Convert price column from string with comma decimal to float
    df['price'] = df['price'].astype(str).str.replace(',', '.').astype(float)

    return df

def give_price(time_index: int) -> float:
    """
    Returns the electricity price at a specific minute index.
    Prices are given hourly in the file, so the value is held constant for 60 minutes.

    Parameters:
    - time_index (int): Minute index (0-based). Example: 0 = 00:00, 60 = 01:00.

    Returns:
    - float: Price in €/Wh (converted from €/MWh by dividing by 1000)
    """
    # Read hourly price data
    price_data = read_energy_prices('data/EnergyPriceManyDays.xlsx')

    if time_index < 0:
        return 0.0

    # Map minute index → hour index
    hour_index = time_index // 60

    if hour_index >= len(price_data):
        return 0.0

    # €/MWh → €/Wh
    return price_data['price'].iloc[hour_index] / 1000