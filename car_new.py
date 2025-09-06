import numpy as np
import pandas as pd
from datetime import datetime

def read_charging_times(csv_path):
    df = pd.read_csv(csv_path, usecols=[
        'Vehicle ID',
        'Charging start time [ISO8601]',
        'Charging end time [ISO8601]',
        'Batt Cap [kWh]',
        'Charger #'
    ])
    return df

def timestamp_to_minute_index(timestamp):
    dt = datetime.fromisoformat(timestamp)
    return dt.hour * 60 + dt.minute



def car_create(csv_path, csv_ports_path):
    df = read_charging_times(csv_path)

    # --- Read charging-port capacities (unique per Ch ID) ---
    df_ports = pd.read_csv(csv_ports_path)

    # Clean columns in case of weird spacing
    df_ports.columns = [c.strip() for c in df_ports.columns]

    # Ensure the expected columns exist
    col_id = 'Ch ID'
    col_p  = 'Max power [kW]'
    if col_id not in df_ports.columns or col_p not in df_ports.columns:
        raise ValueError(f"Port file must contain columns '{col_id}' and '{col_p}'")

    # Group by Ch ID and pick a single capacity per port (max is safe)
    ports_series = (
        df_ports.groupby(col_id)[col_p]
        .max()                         # or .first() if you prefer
        .sort_index()
    )

    # Build a dense list indexed by 1..max_id, fill gaps with 0
    max_id = int(ports_series.index.max())
    charging_port_capacity = [0.0] * max_id
    for ch_id, p_kw in ports_series.items():
        idx = int(ch_id) - 1                 # 1-based → 0-based
        charging_port_capacity[idx] = float(p_kw) * 1000.0  # kW → W

    # ---- rest of your existing code ----
    ID_array = []
    availability = []
    battery_capacity = []
    ID_cap = []
    status = []
    charger_usage = []

    for _, row in df.iterrows():
        vehicle_id  = row['Vehicle ID']
        start_time  = row['Charging start time [ISO8601]']
        end_time    = row['Charging end time [ISO8601]']
        battery_cap = row['Batt Cap [kWh]'] * 1000  # Wh
        charger_num = int(row['Charger #'])

        if vehicle_id not in ID_array:
            ID_array.append(vehicle_id)
            availability.append([2] * 1440)       # 2 = away
            charger_usage.append([0] * 1440)      # 0 = no charger in use
            battery_capacity.append([0] * 1440)
            status.append([0] * 1440)
            ID_cap.append(battery_cap)

        vehicle_index = ID_array.index(vehicle_id)

        start_idx = timestamp_to_minute_index(start_time)
        end_idx   = min(timestamp_to_minute_index(end_time), 1440)

        for i in range(start_idx, end_idx):
            availability[vehicle_index][i] = 3    # charging
            charger_usage[vehicle_index][i] = charger_num

    return (
        ID_array, ID_cap, charger_usage,
        charging_port_capacity, availability,
        battery_capacity, status
    )

ID_array, ID_cap, charger_usage,charging_port_capacity, availability, battery_capacity, status = car_create('data/data_cars.csv','data/port_capacity_car.csv')
