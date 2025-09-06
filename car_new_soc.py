import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yaml

with open("configurations/controller_add_config.yaml", "r") as f:
    config_data = yaml.safe_load(f)

configs = config_data["InitializationSettings"]["configs"][0]

delta_t = configs["timestep"]

def read_charging_times(csv_path):
    df = pd.read_csv(csv_path, usecols=[
        'Vehicle ID',
        'Charging start time [ISO8601]',
        'Charging end time [ISO8601]',   # kept but not used for timing
        'Batt Cap [kWh]',
        'Charger #',                     # ← missing comma fixed
        'Charged SoC [%]',
    ])
    return df

def timestamp_to_minute_index(timestamp):
    dt = datetime.fromisoformat(timestamp)
    return dt.hour * 60 + dt.minute

def minute_index_to_timeiso(day_start_dt, minute_index):
    minute_index = max(0, min(1440, int(minute_index)))
    return (day_start_dt + timedelta(minutes=minute_index)).isoformat(timespec='minutes')

def car_create_soc(csv_path, csv_ports_path, day_start="2024-01-01T00:00", eta = delta_t):
    """
    Compute a NEW end time so the car charges from SoC=0 up to 'Charged SoC [%]'.
    Energy to add = Batt Cap [kWh] * Charged SoC [%]/100.
    Charging duration = ceil( (Wh_to_add / (P_charger_W * eta)) * 60 ).
    Returns:
      ID_array, ID_cap, charger_usage, charging_port_capacity, availability, battery_capacity, status
    """
    df = read_charging_times(csv_path)

    # --- Read charging-port capacities (unique per Ch ID) ---
    df_ports = pd.read_csv(csv_ports_path)
    df_ports.columns = [c.strip() for c in df_ports.columns]
    col_id = 'Ch ID'
    col_p  = 'Max power [kW]'
    if col_id not in df_ports.columns or col_p not in df_ports.columns:
        raise ValueError(f"Port file must contain columns '{col_id}' and '{col_p}'")

    ports_series = df_ports.groupby(col_id)[col_p].max().sort_index()

    # Build dense list indexed by 1..max_id (kW→W)
    max_id = int(ports_series.index.max())
    charging_port_capacity = [0.0] * max_id
    for ch_id, p_kw in ports_series.items():
        charging_port_capacity[int(ch_id) - 1] = float(p_kw) * 1000.0

    # ---- Output arrays ----
    ID_array = []
    availability = []
    battery_capacity = []
    ID_cap = []
    status = []
    charger_usage = []

    day_start_dt = datetime.fromisoformat(day_start)

    for _, row in df.iterrows():
        vehicle_id   = row['Vehicle ID']
        start_time   = row['Charging start time [ISO8601]']   # ISO string
        # original_end = row['Charging end time [ISO8601]']   # not used
        batt_cap_wh  = float(row['Batt Cap [kWh]']) * 1000.0  # Wh (full capacity)
        target_soc_p = float(row['Charged SoC [%]'])          # %
        charger_num  = int(row['Charger #'])

        # Energy to add to reach the requested SoC (from SoC=0)
        energy_needed_wh = batt_cap_wh * (target_soc_p / 100.0)

        # Charger power (W)
        power_w = 0.0
        if 1 <= charger_num <= len(charging_port_capacity):
            power_w = charging_port_capacity[charger_num - 1]

        # Ensure new vehicle row in arrays
        if vehicle_id not in ID_array:
            ID_array.append(vehicle_id)
            availability.append([2] * 1440)       # 2 = away
            charger_usage.append([0] * 1440)      # 0 = no charger in use
            battery_capacity.append([0.0] * 1440) # SoC trace placeholder (update later in EMS)
            status.append([0] * 1440)             # 0 away, 1 charging, 5 not-charging
            ID_cap.append(batt_cap_wh)            # store full capacity in Wh

        v_idx = ID_array.index(vehicle_id)

        # Compute new end time from energy_needed and charger power
        start_idx = timestamp_to_minute_index(start_time)
        if power_w <= 0.0 or energy_needed_wh <= 0.0:
            end_idx = start_idx  # no charging possible
        else:
            duration_minutes = math.ceil((energy_needed_wh / (power_w * eta)) * 60.0)
            end_idx = min(start_idx + duration_minutes, 1440)

        # Fill arrays for the computed charging window
        for i in range(start_idx, end_idx):
            availability[v_idx][i]  = 3          # charging/present
            charger_usage[v_idx][i] = charger_num
            status[v_idx][i]        = 1          # actively charging

        # (Optional) if you want to see the computed end time:
        # end_iso = minute_index_to_timeiso(day_start_dt, end_idx)
        # print(vehicle_id, start_time, '→', end_iso, f'({duration_minutes} min)')

    return (
        ID_array, ID_cap, charger_usage,
        charging_port_capacity, availability,
        battery_capacity, status
    )


