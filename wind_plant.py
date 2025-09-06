import os
# wind_plant.py
# ------------------
# This script models wind energy output based on user-defined wind turbine configurations
# and real weather data using the `windpowerlib` package.
#
# 💡 Functionality:
# - Loads turbine configuration from a YAML file
# - Reads and resamples hourly weather data
# - Constructs a wind farm using WindTurbine objects
# - Computes wind power output for a given time step
#
# 📂 Input:
# - ./data/weather.csv — Weather data with wind speeds, pressure, etc.
# - ./configurations/turbine_config.yaml — User-defined wind turbine parameters
#
# 📤 Output:
# - `power_wind(t)` — Returns wind farm output at hour t (0–479) in watts
import numpy as np
import yaml
import requests
import pandas as pd
from windpowerlib import WindTurbine, WindFarm, TurbineClusterModelChain

# --- Module-level variables to cache weather and wind farm setup ---
_weather_data = None
_wind_farm = None
import pandas as pd

import pandas as pd

def get_weather_data(file_20='windatlas_20.csv', file_100='windatlas_100.csv'):
    """
    Loads wind speed data from two CSVs (height 20 and 100),
    and constructs a multi-index DataFrame with dummy pressure, temperature, and roughness_length,
    where columns have a MultiIndex: (variable_name, height).

    Returns:
        pd.DataFrame: Hourly weather data with shape (480, 7)
    """
    # Read CSVs and skip metadata row
    df_20 = pd.read_csv(file_20, skiprows=1)
    df_100 = pd.read_csv(file_100, skiprows=1)

    # Parse datetime and set as index
    df_20['datetime'] = pd.to_datetime(df_20['datetime'])
    df_100['datetime'] = pd.to_datetime(df_100['datetime'])

    df_20.set_index('datetime', inplace=True)
    df_100.set_index('datetime', inplace=True)

    # Rename columns to multi-index format
    df_20.columns = pd.MultiIndex.from_tuples([('wind_speed', 20)])
    df_100.columns = pd.MultiIndex.from_tuples([('wind_speed', 100)])

    # Combine wind speeds
    df = pd.concat([df_20, df_100], axis=1)

    # Add dummy temperature and pressure values
    df[('temperature', 2)] = 0
    df[('temperature', 80)] = 0
    df[('temperature', 10)] = 0  # ✅ Added this to prevent KeyError
    df[('pressure', 0)] = 0
    df[('pressure', 10)] = 0

    # Add roughness_length at height 0
    df[('roughness_length', 0)] = 0.15

    # Sort columns to match desired order
    df = df[[('pressure', 0), ('temperature', 2), ('wind_speed', 20), ('roughness_length', 0),
             ('temperature', 10), ('wind_speed', 100), ('temperature', 80)]]

    # Localize and convert time
    df.index = df.index.tz_localize('UTC').tz_convert('Europe/Berlin')

    # Final formatting
    df.columns.names = ['variable_name', 'height']
    return df.resample('1h').first().iloc[:480]

def load_turbine_config(path='configurations/turbine_config.yaml'):

    """
    Load wind turbine configuration from a YAML file.

    Args:
        path (str): Path to turbine config file

    Returns:
        list[dict]: List of turbine configuration dictionaries
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config["wind_turbines"]


def collect_turbines(turbine_data: list):

    """
    Build a turbine fleet DataFrame from configuration list.

    Args:
        turbine_data (list): List of turbine dictionaries

    Returns:
        pd.DataFrame: DataFrame of WindTurbine objects and their counts
    """
    fleet = []
    for i, t in enumerate(turbine_data):
        turbine = WindTurbine(
            name=f"turbine_{i}",
            hub_height=t["hub_height"],
            turbine_type=t["turbine_id"],
        )
        fleet.append({
            "wind_turbine": turbine,
            "number_of_turbines": float(t["number_of_turbines"]),
            "total_capacity": np.nan
        })
    return pd.DataFrame(fleet)[["wind_turbine", "number_of_turbines", "total_capacity"]]


def init_wind_farm():

    """
    Initializes and caches the wind farm simulation model.

    Loads:
    - Weather data
    - Turbine configuration

    Then:
    - Builds turbine fleet
    - Initializes global WindFarm object
    """
    global _weather_data, _wind_farm
    if _weather_data is None:
        _weather_data = get_weather_data()

    turbine_config = load_turbine_config()
    turbine_fleet = collect_turbines(turbine_config)

    _wind_farm = WindFarm(wind_turbine_fleet=turbine_fleet, name='UserDefinedFarm')

init_wind_farm()

def power_wind(time_index: int) -> float:
    """
    Returns wind farm power output at a given minute index.

    Weather data is hourly, so each hour's power is repeated for 60 minutes.

    Args:
        time_index (int): Minute index (0-based). Example: 0 = 00:00, 60 = 01:00.

    Returns:
        float: Total wind farm power output in watts.
    """
    global _weather_data, _wind_farm

    if time_index < 0:
        return 0.0

    # map minute index to hourly weather index
    hour_index = time_index // 60

    if hour_index >= len(_weather_data):
        return 0.0

    weather_step = _weather_data.iloc[[hour_index]]
    model_chain = TurbineClusterModelChain(_wind_farm)
    model_chain.run_model(weather_step)

    return model_chain.power_output.iloc[0].sum()