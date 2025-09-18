# car_new.py
import yaml
import pandas as pd
import numpy as np

def _to_iso(date_str: str, time_str: str) -> str:
    return f"{str(date_str).strip()} {str(time_str).strip()}"

def _read_ports_from_yaml(ports_yaml_path: str) -> list[float]:
    with open(ports_yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}
    ports = (data.get("InitializationSettings") or {}).get("ports") or data.get("ports") or []
    max_id, parsed = 0, []
    for p in ports:
        pid = int(p.get("port_id", 0))
        if pid <= 0:
            continue
        if p.get("capacity_W") is not None:
            cap_w = float(p["capacity_W"])
        elif p.get("capacity_kW") is not None:
            cap_w = float(p["capacity_kW"]) * 1000.0
        else:
            cap_w = 0.0
        parsed.append((pid, cap_w))
        max_id = max(max_id, pid)
    capacities = [0.0] * max(1, max_id)
    for pid, cap_w in parsed:
        capacities[pid - 1] = cap_w
    return capacities

def _timestamp_to_step(ts, base0, start_minutes, delta_minutes):
    dt = pd.to_datetime(str(ts), errors="coerce")
    if pd.isna(dt):
        return 0
    rel_min = int((dt - base0).total_seconds() // 60) - int(start_minutes)
    if rel_min < 0:
        return 0
    return rel_min // int(delta_minutes)

def car_df_from_yaml(ev_yaml_path: str) -> pd.DataFrame:
    with open(ev_yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cars = (cfg.get("InitializationSettings") or {}).get("cars", [])
    rows = []
    for i, car in enumerate(cars):
        vid = str(car.get("id", f"Vehicle#{i+1}"))
        cap_wh = float(car.get("battery_capacity", 0.0))
        batt_kwh = cap_wh / 1000.0
        rows.append({
            "Vehicle ID": vid,
            "Charging start time [ISO8601]": _to_iso(car.get("arrival_date",""),   car.get("arrival_time","00:00")),
            "Charging end time [ISO8601]":   _to_iso(car.get("departure_date",""), car.get("departure_time","00:00")),
            "Batt Cap [kWh]": batt_kwh,
            "Charger #": int(car.get("charging_port", 0)),
        })
    return pd.DataFrame(rows)

def car_create_from_yaml(
    ev_yaml_path: str,
    ports_yaml_path: str,
    *,
    start_minutes: int,       # <<<< from main loop
    delta_minutes: int,       # <<<< from main loop
    steps_len: int            # <<<< from main loop
):
    df = car_df_from_yaml(ev_yaml_path)
    port_caps = _read_ports_from_yaml(ports_yaml_path)

    ID_array, availability, battery_capacity = [], [], []
    ID_cap, status, charger_usage = [], [], []

    # base0 for multi-day alignment = earliest midnight in the file
    ts_all = pd.to_datetime(
        pd.concat([df["Charging start time [ISO8601]"], df["Charging end time [ISO8601]"]], ignore_index=True),
        errors="coerce"
    ).dropna()
    base0 = (ts_all.min().normalize() if not ts_all.empty else pd.Timestamp("1970-01-01"))

    for _, row in df.iterrows():
        vid   = str(row["Vehicle ID"])
        start = row["Charging start time [ISO8601]"]
        end   = row["Charging end time [ISO8601]"]
        cap_w = float(row["Batt Cap [kWh]"]) * 1000.0
        port  = int(row["Charger #"])

        if vid not in ID_array:
            ID_array.append(vid)
            availability.append([2] * steps_len)       # 2 = away
            charger_usage.append([0] * steps_len)      # 0 = not on any port
            battery_capacity.append([0.0] * steps_len) # Wh timeline (updated by controller)
            status.append([0] * steps_len)
            ID_cap.append(cap_w)

        idx = ID_array.index(vid)
        s = _timestamp_to_step(start, base0, start_minutes, delta_minutes)
        e = _timestamp_to_step(end,   base0, start_minutes, delta_minutes)
        s = max(0, min(s, steps_len))
        e = max(s, min(e, steps_len))

        for t in range(s, e):
            availability[idx][t]  = 3   # at charger
            charger_usage[idx][t] = port

    return (
        ID_array,           # car_names
        ID_cap,             # battery_caps (Wh)
        charger_usage,      # which physical port at each step
        port_caps,          # port power ratings [W]
        availability,       # 2 away / 3 at charger
        battery_capacity,   # Wh trajectory (filled by controller)
        status              # status per step (0/1/5), filled in main loop
    )
