# load_profile.py
# ----------------------
# Utilities to:
# 1) Read minute-level load (kW) from Excel/CSV file with ISO8601 timestamps
# 2) Retrieve the load at a given timestep (with custom step size)
#
# Columns expected:
#   - "Time [ISO8601]" (datetime string)
#   - "Consumption without charging [kW]" (float)

import pandas as pd

def read_load_profile(filepath: str) -> pd.DataFrame:
    """
    Reads the CSV file and returns a DataFrame with 'time' and 'load_kw' columns.
    """
    df = pd.read_csv(filepath, usecols=["Time [ISO8601]", "Consumption without charging [kW]"])

    # Rename columns
    df.columns = ["time", "load_kw"]

    # Parse datetime
    df["time"] = pd.to_datetime(df["time"])
    df["load_kw"] = df["load_kw"].astype(float)

    return df



def give_load_w(time_index: int, delta: int) -> float:
    """
    Returns the electrical load (Watts) at a specific timestep.

    Parameters:
    - time_index (int): Simulation step index (0-based)
    - delta (int): Timestep size in minutes (e.g., 1 = 1-min resolution, 15 = quarter-hourly steps)
    - filepath (str): Path to the load data file

    Returns:
    - float: Load in Watts
    """
    filepath = 'data/LoadManyDays.xlsx'
    df = read_load_profile(filepath)

    # Row index based on timestep size
    row_idx = time_index * delta

    if row_idx >= len(df):
        return 0.0

    return float(df.loc[row_idx, "load_kw"]) * 1000.0  # convert kW → W
