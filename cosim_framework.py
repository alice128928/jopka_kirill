# cosim.py
# Fully Δt/end_time flexible co-simulation + interactive plot selector

import os
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy import shape

from load_reading import give_load_w
from ev_request import ev_generate_from_config  # (kept for compatibility, not used directly here)
from car_new import car_create_from_yaml
      # (kept for compatibility)
from price_market import give_price
from battery_storage import BatteryStorage

SESSIONS_CSV = "data/data_cars.csv"   # optional sessions file with Initial SoC etc.


# ----------------- YAML → sessions fallback (for SoC) -----------------
def _sessions_df_from_yaml(ev_yaml_path: str) -> pd.DataFrame:
    """Build sessions from configurations/ev_config.yaml if CSV is absent."""
    try:
        with open(ev_yaml_path, "r") as f:
            y = yaml.safe_load(f)
    except Exception:
        y = None

    cars = (y or {}).get("InitializationSettings", {}).get("cars", []) or []
    rows = []
    for row in cars:
        vid = str(row.get("id", "Vehicle#"))
        ad  = str(row.get("arrival_date", ""))     # 'YYYY-MM-DD'
        at  = str(row.get("arrival_time", ""))     # 'HH:MM' or 'HH:MM:SS'
        dd  = str(row.get("departure_date", ""))
        dtm = str(row.get("departure_time", ""))

        def _combine(d, t):
            try:
                if len(t) == 5:
                    hhmm = datetime.strptime(t, "%H:%M").time()
                else:
                    hhmm = datetime.strptime(t, "%H:%M:%S").time()
                iso = datetime.combine(datetime.fromisoformat(d).date(), hhmm).isoformat()
                return iso
            except Exception:
                return f"{d}T{t}" if d and t else ""

        eta_iso = _combine(ad, at)
        etd_iso = _combine(dd, dtm)

        rows.append({
            "Vehicle ID": vid,
            "ETA [ISO8601]": eta_iso,
            "ETD [ISO8601]": etd_iso,
            "Charging start time [ISO8601]": eta_iso,  # start = ETA
            "Charging end time [ISO8601]": etd_iso,    # end   = ETD
            "Initial SoC [%]": 0.0,                    # if not provided
        })

    return pd.DataFrame(rows)


# ----------------- misc helpers -----------------
def _resolve_car_idx(names, wanted_id: str) -> int:
    if not names:
        return 0
    wanted_id = (wanted_id or "").strip()
    for i, nm in enumerate(names):
        if str(nm).strip() == wanted_id:
            return i
    return 0


def port_power_series(charger_usage, charging_ports, selected_port, num_steps):
    """Return [W] series for selected_port, safe for any horizon."""
    n_ports = len(charging_ports) if charging_ports is not None else 0
    if n_ports == 0:
        return [0.0] * num_steps

    selected_port = int(selected_port)
    if selected_port < 1 or selected_port > n_ports:
        selected_port = 1

    n_cars = len(charger_usage) if charger_usage is not None else 0
    avail_T = min((len(charger_usage[i]) for i in range(n_cars)), default=0)

    out = []
    for t in range(num_steps):
        if t < avail_T:
            active = any(int(charger_usage[c][t]) == selected_port for c in range(n_cars))
            out.append(float(charging_ports[selected_port - 1]) if active else 0.0)
        else:
            out.append(0.0)
    return out


def _read_sessions_dataframe(path: str) -> pd.DataFrame:
    """Try CSV first; if missing/invalid, fallback to YAML sessions."""
    cols = [
        "Vehicle ID", "ETA [ISO8601]", "ETD [ISO8601]",
        "Charging start time [ISO8601]", "Charging end time [ISO8601]",
        "Initial SoC [%]"
    ]

    use_yaml_fallback = False
    df = pd.DataFrame(columns=cols)

    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip() for c in df.columns]
            if not set(cols[:5]).issubset(df.columns):
                use_yaml_fallback = True
            else:
                if "Initial SoC [%]" not in df.columns:
                    df["Initial SoC [%]"] = 0.0
        except Exception:
            use_yaml_fallback = True
    else:
        use_yaml_fallback = True

    if use_yaml_fallback:
        df = _sessions_df_from_yaml("configurations/ev_config.yaml")

    for col in cols[1:5]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Initial SoC [%]" not in df.columns:
        df["Initial SoC [%]"] = 0.0

    return df[cols] if set(cols).issubset(df.columns) else df


# ---------- Δt + multi-day aware mapping for sessions ----------
def _compute_base0_for_sessions(sdf: pd.DataFrame) -> pd.Timestamp:
    all_ts = pd.to_datetime(pd.concat([
        sdf["ETA [ISO8601]"], sdf["Charging start time [ISO8601]"],
        sdf["Charging end time [ISO8601]"], sdf["ETD [ISO8601]"]
    ], ignore_index=True), errors="coerce").dropna()
    return (all_ts.min().normalize() if not all_ts.empty else pd.Timestamp("1970-01-01"))


def _ts_to_step_index(ts, base0: pd.Timestamp, start_minutes: int, delta_t: int) -> int:
    dt = pd.to_datetime(str(ts), errors="coerce")
    if pd.isna(dt):
        return 0
    rel_min = int((dt - base0).total_seconds() // 60) - int(start_minutes)
    if rel_min < 0:
        return 0
    return rel_min // max(1, int(delta_t))


def _sessions_for_vehicle_step_index(
    sdf: pd.DataFrame, vehicle_id: str, steps_len: int, start_minutes: int, delta_t: int
) -> list[dict]:
    need = [
        "Vehicle ID","ETA [ISO8601]","ETD [ISO8601]",
        "Charging start time [ISO8601]","Charging end time [ISO8601]","Initial SoC [%]"
    ]
    for c in need:
        if c not in sdf.columns:
            raise ValueError(f"Sessions CSV missing column: {c}")

    df = sdf[sdf["Vehicle ID"].astype(str).str.strip() == str(vehicle_id).strip()].copy()
    if df.empty:
        return []

    df.sort_values("ETA [ISO8601]", inplace=True)
    base0 = _compute_base0_for_sessions(df)

    out = []
    for _, row in df.iterrows():
        eta_i   = _ts_to_step_index(row["ETA [ISO8601]"], base0, start_minutes, delta_t)
        start_i = _ts_to_step_index(row["Charging start time [ISO8601]"], base0, start_minutes, delta_t)
        end_i   = _ts_to_step_index(row["Charging end time [ISO8601]"], base0, start_minutes, delta_t)
        etd_i   = _ts_to_step_index(row["ETD [ISO8601]"], base0, start_minutes, delta_t)

        eta_i   = max(0, min(eta_i,   steps_len - 1))
        start_i = max(start_i, eta_i);  start_i = min(start_i, steps_len - 1)
        end_i   = max(end_i,   start_i); end_i = min(end_i,   steps_len - 1)
        etd_i   = max(etd_i,   end_i);   etd_i = min(etd_i,   steps_len - 1)

        out.append({
            "eta_i": eta_i, "start_i": start_i, "end_i": end_i, "etd_i": etd_i,
            "eta_ts": row["ETA [ISO8601]"], "start_ts": row["Charging start time [ISO8601]"],
            "end_ts": row["Charging end time [ISO8601]"], "etd_ts": row["ETD [ISO8601]"],
            "initial_soc": float(row.get("Initial SoC [%]", np.nan)),
        })
    return out


# -------- SoC from capacity peaks within each session (Δt-safe) --------
def soc_timeline_from_session_peaks(
    car_idx: int, car_id: str, battery_capacity_arrays, battery_caps,
    charger_usage, steps_len: int, sessions_df_path: str,
    start_minutes: int, delta_t: int
) -> np.ndarray:

    cap_wh_total = float(battery_caps[car_idx])
    cap_series = np.asarray(battery_capacity_arrays[car_idx], dtype=float)
    ports = np.asarray(charger_usage[car_idx], dtype=int) if charger_usage is not None else np.zeros_like(cap_series, int)

    T = min(int(steps_len), len(cap_series), len(ports))
    if T <= 0 or cap_wh_total <= 0:
        return np.array([])

    cap = cap_series[:T]
    sessions_raw = _read_sessions_dataframe(sessions_df_path)
    sessions = _sessions_for_vehicle_step_index(sessions_raw, car_id, T, start_minutes, delta_t)

    soc = np.zeros(T, dtype=float)

    if not sessions:
        first_soc = (cap[0] / cap_wh_total) * 100.0
        soc[:] = first_soc
        print(f"[{car_id}] No sessions found; SoC held at {first_soc:.2f}%")
        return soc

    def _fmt(ts):
        try:
            return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(ts)

    print(f"\nSession audit for {car_id} (ΔSoC added per charging window)")
    total_bump = 0.0

    first = sessions[0]
    hold = float(first["initial_soc"]) if np.isfinite(first["initial_soc"]) else (cap[0] / cap_wh_total) * 100.0
    soc[:first["eta_i"]] = hold
    last_etd = 0

    for idx, s in enumerate(sessions, start=1):
        eta_i, st_i, en_i, etd_i = s["eta_i"], s["start_i"], s["end_i"], s["etd_i"]
        init_soc = float(s["initial_soc"]) if np.isfinite(s["initial_soc"]) else hold

        if last_etd < eta_i:
            soc[last_etd:eta_i] = hold

        if eta_i < st_i:
            soc[eta_i:st_i] = init_soc
        else:
            st_i = eta_i

        if en_i >= st_i:
            peak_wh = float(cap[st_i:en_i+1].max())
            bump_pct = (peak_wh / cap_wh_total) * 100.0
            final_soc = init_soc + bump_pct

            n = max(1, en_i - st_i + 1)
            ramp = np.linspace(init_soc, final_soc, n)
            soc[st_i:en_i+1] = ramp
        else:
            bump_pct = 0.0
            final_soc = init_soc
            peak_wh = 0.0

        if en_i < etd_i:
            soc[en_i:etd_i] = final_soc

        print(f"[{car_id}] Session {idx}: {_fmt(s['start_ts'])}→{_fmt(s['end_ts'])} | "
              f"Initial={init_soc:.5f}% | ΔSoC={bump_pct:.2f}% | Final={final_soc:.2f}% "
              f"(peak {peak_wh:.0f} Wh)")
        total_bump += bump_pct

        hold = final_soc
        last_etd = etd_i

    if last_etd < T:
        soc[last_etd:] = hold

    print(f"[{car_id}] Total ΔSoC added across sessions: {total_bump:.2f}%\n")
    return soc


# ---------- light wrapper ----------
class Model:
    def __init__(self, process_model):
        if not callable(process_model):
            raise ValueError("The process must be a function or callable class.")
        self.process_model = process_model

    def calculate(self, *args):
        return self.process_model(*args)


class Manager:
    def __init__(self, models: list[Model], settings_configuration: dict):
        self.models = models
        self.electric_grid = models[0]
        self.controller = models[1]
        self.wind = models[2]
        self.solar = models[3]
        self.evstate = models[4]
        self.price_market = models[5]
        self.storage_battery = models[6]
        self.controller_2 = models[-1]
        self.settings_configuration = settings_configuration

    def run_simulation(self):
        # ---- timing + sim config ----
        config = self.settings_configuration
        config_id  = config["InitializationSettings"]["config_id"]
        start_time = int(config["InitializationSettings"]["time"]["start_time"])  # minutes
        end_time   = int(config["InitializationSettings"]["time"]["end_time"])    # minutes

        with open("configurations/controller_add_config.yaml", "r") as f:
            config_data = yaml.safe_load(f)

        configs = config_data["InitializationSettings"]["configs"][0]
        grid_capacity     = float(configs["grid_capacity"])
        initial_money     = float(configs["initial_money"])
        storage_capacity  = float(configs["storage_capacity"])
        delta_t           = int(configs["timestep"])
        selected_port     = int(configs.get("plot_port", 1))
        selected_car_id   = str(configs.get("plot_car_id", "")).strip()

        # ---- accumulators ----
        times = []  # minutes since sim start (x-axis)
        wind_energy = []
        solar_energy = []
        power_setpoint_array = [0]          # one element longer than times
        storage_array = []
        storage_array_percentage = []
        money_array = [initial_money]
        load_array = []
        market_price_array = []

        # ---- storage battery object ----
        battery = self.storage_battery.calculate(storage_capacity, 0.95, 0.95, 0)

        steps_len = max(0, (end_time - start_time) // max(1, delta_t))

        (car_names, battery_caps, charger_usage, charging_ports,
         availability_arrays, battery_capacity_arrays, status_cars) = car_create_from_yaml(
            ev_yaml_path="configurations/ev_config.yaml",
            ports_yaml_path="configurations/port_data.yaml",
            start_minutes=start_time,
            delta_minutes=delta_t,
            steps_len=steps_len,
        )
        car_idx_to_plot = _resolve_car_idx(car_names, selected_car_id)
        print(status_cars)
        print(battery_caps)
        print(car_names)
        print(charger_usage)
        print(charging_ports)
        # ---- horizon built only from start/end/Δt ----
        time_steps = max(0, (end_time - start_time) // max(1, delta_t))

        # ---- main loop ----
        for step in range(time_steps):
            tmin = start_time + step * delta_t  # minutes since sim zero
            time_clock = tmin

            wind_energy_time   = self.wind.calculate(time_clock)
            solar_energy_time  = self.solar.calculate(time_clock)
            current_price_time = self.price_market.calculate(time_clock)
            ev_state_time      = self.evstate.calculate(step, availability_arrays, battery_caps, battery_capacity_arrays)

            constant_load_time = give_load_w(time_clock)

            for i in range(len(status_cars)):
                status_cars[i][step] = ev_state_time[i]

            money, battery, power_set_point, battery_capacity_arrays, _ = self.controller.calculate(
                status_cars, charging_ports, battery, money_array[step], step, battery_capacity_arrays,
                solar_energy_time, wind_energy_time, current_price_time, storage_capacity, grid_capacity,
                0, power_setpoint_array[step],
                battery_caps, availability_arrays, charger_usage, delta_t, constant_load_time
            )

            # accumulate
            times.append(tmin)
            wind_energy.append(wind_energy_time)
            solar_energy.append(solar_energy_time)
            power_setpoint_array.append(power_set_point)
            storage_array.append(battery.get_soc())
            storage_array_percentage.append(battery.get_soc_percentage())
            money_array.append(money)
            load_array.append(constant_load_time)
            market_price_array.append(current_price_time)

        # ---- derived series ----
        selected_port_power = port_power_series(
            charger_usage=charger_usage,
            charging_ports=charging_ports,
            selected_port=selected_port,
            num_steps=len(times),
        )

        soc_pct = soc_timeline_from_session_peaks(
            car_idx_to_plot,
            car_names[car_idx_to_plot] if car_names else "",
            battery_capacity_arrays,
            battery_caps,
            charger_usage,
            steps_len=len(times),
            sessions_df_path=SESSIONS_CSV,
            start_minutes=start_time,
            delta_t=delta_t,
        )

        # align setpoints to times
        ps_series = np.asarray(power_setpoint_array[1:1 + len(times)], dtype=float)

        # ---- plot ----
        results = self.plot_results_pred(
            np.asarray(times),
            selected_port_power,
            solar_energy,
            wind_energy,
            storage_array,
            storage_array_percentage,
            ps_series,
            money_array,
            config_id,
            charger_usage,
            charging_ports,
            car_idx_to_plot,
            selected_port,
            soc_pct,
        )

        return results

    def plot_results_pred(
        self,
        times,
        port_power,
        solar_energy,
        wind_energies,
        storage_array,
        storage_array_percentage,
        power_setpoint_series,
        money_array,
        config_id,
        charger_usage,
        power_charging_port,
        car_idx_to_plot,
        selected_port,
        soc_pct,
    ):
        # ---- Trim to a common length (SoC is separate timeline but same steps) ----
        series_1d = [
            port_power, solar_energy, wind_energies,
            storage_array, storage_array_percentage,
            power_setpoint_series, money_array
        ]
        min_len = min([len(times)] + [len(s) for s in series_1d]) if len(series_1d) else len(times)
        times = np.asarray(times)[:min_len]
        port_power = np.asarray(port_power)[:min_len]
        solar_energy = np.asarray(solar_energy)[:min_len]
        wind_energies = np.asarray(wind_energies)[:min_len]
        storage_array = np.asarray(storage_array)[:min_len]
        storage_array_percentage = np.asarray(storage_array_percentage)[:min_len]
        power_setpoint_series = np.asarray(power_setpoint_series)[:min_len]
        money_array = np.asarray(money_array)[:min_len]
        index = car_idx_to_plot + 1

        # SoC timeline — same horizon
        soc_plot = np.asarray(soc_pct)[:min_len]

        # ---- Build a single-axes figure with 8 traces (one visible at a time) ----
        fig = go.Figure()

        # X-axis is "minutes since start" everywhere
        traces = [
            dict(name=f"Power at Port #{selected_port}",
                 x=times, y=port_power,
                 ylab="Power [W] usage of the selected port",
                 xlab="Minutes since start"),
            dict(name=f"SoC Timeline of car #{index}",
                 x=times, y=soc_plot,
                 ylab="SoC [%] of the selected car",
                 xlab="Minutes since start"),
            dict(name="Solar panels energy production",
                 x=times, y=solar_energy,
                 ylab="Solar Energy [W]",
                 xlab="Minutes since start"),
            dict(name="Wind turbine energy production",
                 x=times, y=wind_energies,
                 ylab="Wind Energy [W]",
                 xlab="Minutes since start"),
            dict(name="Storage (Energy) at the hub",
                 x=times, y=storage_array,
                 ylab="Storage [Wh]",
                 xlab="Minutes since start"),
            dict(name="Storage (State of Charge %) at the hub",
                 x=times, y=storage_array_percentage,
                 ylab="SoC [%] of the energy storage",
                 xlab="Minutes since start"),
            dict(name="Operational cost of the hub",
                 x=times, y=money_array,
                 ylab="Money [€]",
                 xlab="Minutes since start"),
            dict(name="Grid perspective overview",
                 x=times, y=power_setpoint_series,
                 ylab="Setpoint [W]",
                 xlab="Minutes since start"),
        ]

        # Add all traces (only the first visible initially)
        for i, t in enumerate(traces):
            fig.add_trace(go.Scatter(
                x=t["x"], y=t["y"], mode="lines",
                name=t["name"],
                visible=(i == 0)
            ))

        # Helper to build visibility mask + axis/title updates
        def _button(i):
            vis = [False] * len(traces)
            vis[i] = True
            lock_soc_range = traces[i]["name"].startswith("Storage (State of Charge")
            return dict(
                label=str(i + 1),
                method="update",
                args=[
                    {"visible": vis},
                    {
                        "title": {"text": traces[i]["name"]},
                        "xaxis": {"title": {"text": traces[i]["xlab"]}},
                        "yaxis": {
                            "title": {"text": traces[i]["ylab"]},
                            **({"range": [0, 100]} if lock_soc_range else {})
                        },
                    },
                ],
            )

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.5, xanchor="center",
                    y=1.12, yanchor="top",
                    buttons=[_button(i) for i in range(len(traces))],
                    showactive=True,
                )
            ],
            title={"text": traces[0]["name"], "x": 0.5, "xanchor": "center"},
            xaxis={"title": {"text": traces[0]["xlab"]}},
            yaxis={"title": {"text": traces[0]["ylab"]}},
            template="plotly_dark",
            height=650,
            width=1100,
            showlegend=False,
            margin=dict(l=80, r=30, t=100, b=80),
        )

        # Save interactive HTML that preserves the selector
        out_html = f"results_config{config_id}_selector.html"
        fig.write_html(out_html)

        return fig
