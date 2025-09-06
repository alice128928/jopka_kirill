import numpy as np
import yaml
from math import ceil

def _time_to_step_index(time_str, delta_minutes):
    """
    Convert 'HH:MM' or int minutes into a step index within a single day,
    given the step size in minutes.
    """
    if isinstance(time_str, (int, float)):
        total_minutes = int(time_str)
    elif isinstance(time_str, str) and ":" in time_str:
        h, m = map(int, time_str.split(":"))
        total_minutes = h * 60 + m
    else:
        # If it's a string like "10", interpret as minutes-in-day
        total_minutes = int(time_str)

    return total_minutes // int(delta_minutes)


def ev_generate_from_config(
    config_path,
    time_steps,
    delta_minutes
):
    """
    Build EV inputs from YAML for an arbitrary horizon.

    Args:
        config_path: path to the YAML with cars list.
        time_steps:  total number of simulation steps in the horizon.
        delta_minutes: minutes per simulation step (e.g., 1, 5, 15, 60).

    Returns:
        car_names                : list[str]
        battery_caps             : list[float]  (Wh)
        charging_ports           : list[float]  (W)
        availability_arrays      : list[np.ndarray] (each length = time_steps; 2=away, 3=home)
        battery_capacity_arrays  : list[np.ndarray] (zeros, length = time_steps)
        status_cars              : list[np.ndarray] (zeros, length = time_steps)
    """
    # Steps in one 24h day for this resolution
    steps_per_day = int((24 * 60) // delta_minutes)
    if steps_per_day <= 0:
        raise ValueError("delta_minutes must be a positive divisor of 1440.")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    cars = (config.get("InitializationSettings", {}) or {}).get("cars", []) or []

    car_names = []
    battery_caps = []
    charging_ports = []
    availability_arrays = []
    battery_capacity_arrays = []
    status_cars = []

    for car in cars:
        car_id          = car["id"]
        battery_capacity_wh = float(car["battery_capacity"])          # expected Wh in config
        charging_port_w = float(car["charging_port"])                 # expected W in config
        arr_step = _time_to_step_index(car["arrival_time"],    delta_minutes)
        dep_step = _time_to_step_index(car["departure_time"],  delta_minutes)

        # 2 = away by default, 3 = at home (available/charging window)
        daily_avail = np.full(steps_per_day, 2, dtype=np.int16)

        # Clamp to valid range just in case
        arr_step = max(0, min(arr_step, steps_per_day - 1))
        dep_step = max(0, min(dep_step, steps_per_day - 1))

        if arr_step < dep_step:
            daily_avail[arr_step:dep_step] = 3
        elif arr_step > dep_step:
            # Wrap over midnight
            daily_avail[arr_step:] = 3
            daily_avail[:dep_step] = 3
        else:
            # Same step → treat as at home for the whole day or at least one step?
            # Here we assume "home all day" if arrival==departure
            daily_avail[:] = 3

        # Tile daily pattern to cover the horizon, then trim to exact length
        repeats = ceil(time_steps / steps_per_day)
        avail_full = np.tile(daily_avail, repeats)[:time_steps]

        # Append outputs
        car_names.append(car_id)
        battery_caps.append(battery_capacity_wh)
        charging_ports.append(charging_port_w)
        availability_arrays.append(avail_full)
        battery_capacity_arrays.append(np.zeros(time_steps, dtype=float))
        status_cars.append(np.zeros(time_steps, dtype=np.int8))

    return (
        car_names,
        battery_caps,
        charging_ports,
        availability_arrays,
        battery_capacity_arrays,
        status_cars,
    )
