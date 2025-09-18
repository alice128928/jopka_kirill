# tab_components.py
import datetime
import os
import streamlit as st
import pandas as pd
import yaml
from config_setup import delete_config, load_config

def start_tab():
    """Landing tab: image + short explanation of the app workflow."""
    import streamlit as st
    import os

    st.header("🚀 Welcome")

    img_path = "power_new_2.png"
    if os.path.exists(img_path):
        # Centered smaller image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img_path, width=620)  # adjust width as needed
    else:
        st.warning(f"Image '{img_path}' not found next to the app. Place it in the project root.")

    st.markdown(
        """
**How this app works (quick tour):**

1. **📁 Files** – Upload YAML/CSV files or load existing ones to populate all tabs automatically.
2. **⚙️ Settings / 🔋 Battery / ⚡ Grid / 🏠 Load** – Tune global simulation parameters, storage, grid, and load.
3. **⚡ Charging Ports / 🚗 EVs** – Define ports and vehicles (sessions, ports, battery capacities).
4. **💨 Wind / ☀️ Solar** – Configure renewable sources and locations.
5. **💾 Save Configuration** – Writes all current values to `configurations/*.yaml`.
6. **▶️ Run Simulation** – Runs with the YAMLs in `configurations/` and produces results.
7. **📊 Results** – See the output figure at the bottom of the page.

> Tip: Start in **Files** to import everything, then tweak in the other tabs if needed.
        """
    )


def parse_load_csv_to_session_state(csv_file):
    """
    Expect columns:
      - 'Time [ISO8601]'
      - 'Consumption without charging [kW]'
    Stores into st.session_state['load_mode'='timeseries'] and ['load_timeseries'].
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_file)

        required = {'Time [ISO8601]', 'Consumption without charging [kW]'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in load CSV: {missing}")

        records = []
        for _, r in df.iterrows():
            t = str(r['Time [ISO8601]'])
            kw = float(r['Consumption without charging [kW]'])
            records.append({"time": t, "load_w": kw * 1000.0})

        st.session_state["load_mode"] = "timeseries"
        st.session_state["load_timeseries"] = records
        return f"Loaded {len(records)} load points from CSV"
    except Exception as e:
        st.error(f"Error parsing load CSV: {e}")
        return None

# ----------------------------- YAML PARSER -----------------------------

def parse_yaml_configs_to_session_state(controller_config, ev_config, solar_config, turbine_config, ports_config=None,load_config=None ):
    """
    Parse uploaded YAML configurations and populate session state with the values.
    This makes the uploaded configs immediately visible and editable in the tabs.
    """

    # Track which configs were loaded
    loaded_configs = []

    # Load profile (can live in controller_config or dedicated load_config)
    def _ingest_load_profile(blob: dict):
        lp = blob.get("load_profile", {})
        mode = lp.get("mode", "constant")
        st.session_state["load_mode"] = mode
        if mode == "constant":
            st.session_state["load"] = float(lp.get("constant_load_w", 0.0))
            st.session_state["load_timeseries"] = []
        elif mode == "timeseries":
            pts = lp.get("points", [])
            # Normalize to [{"time": ISO8601 str, "load_w": float}]
            norm = []
            for p in pts:
                t = p.get("time") or p.get("timestamp") or p.get("Time [ISO8601]") or p.get("Time")
                v = p.get("load_w", p.get("load_W", p.get("load_kw", p.get("Consumption without charging [kW]"))))
                if v is not None and "kW" in (p.get("unit", "").lower() or ""):
                    v = float(v) * 1000.0
                if "Consumption without charging [kW]" in p:
                    v = float(p["Consumption without charging [kW]"]) * 1000.0
                norm.append({"time": str(t), "load_w": float(v or 0.0)})
            st.session_state["load_timeseries"] = norm

    # From controller_config
    if controller_config and isinstance(controller_config, dict):
        # legacy constant from controller
        if "InitializationSettings" in controller_config:
            cfg0 = controller_config["InitializationSettings"].get("configs", [])
            if cfg0:
                st.session_state["load"] = cfg0[0].get("add_load", st.session_state.get("load", 0.0))
        _ingest_load_profile(controller_config)

    # From dedicated load_config.yaml (if supplied)
    if load_config and isinstance(load_config, dict):
        _ingest_load_profile(load_config)

    # Parse controller config (system settings)
    if controller_config and "InitializationSettings" in controller_config:
        configs = controller_config["InitializationSettings"].get("configs", [])
        if configs:
            config = configs[0]  # Use first config

            # System settings
            st.session_state["money"] = config.get("initial_money", 0.0)
            st.session_state["price_high"] = config.get("price_high", 0.0)
            st.session_state["price_low"] = config.get("price_low", 0.0)
            st.session_state["timestep"] = config.get("timestep", 0)
            st.session_state["config_name"] = controller_config["InitializationSettings"].get("config_id", "")

            # Storage and grid
            st.session_state["storage"] = config.get("storage_capacity", 0.0)
            st.session_state["grid"] = config.get("grid_capacity", 0.0)

            # Load settings (constant load)
            st.session_state["load"] = config.get("add_load", 0.0)

            # Plot settings
            st.session_state["plot_port"] = config.get("plot_port", 1)
            st.session_state["plot_car_id"] = config.get("plot_car_id", "")

            loaded_configs.append("System Settings (Settings, Battery, Grid, Load tabs)")

    # Parse EV config  —— supports flattened-per-session OR multi-session by grouping on id
    if ev_config and "InitializationSettings" in ev_config:
        cars = ev_config["InitializationSettings"].get("cars", [])
        if cars:
            from collections import defaultdict
            import datetime as _dt

            # Build a quick lookup for port labels (if we already have ports in session)
            ports = st.session_state.get("ports", [])
            port_label = {
                p.get("port_id", i + 1): f"Port {p.get('port_id', i + 1)} — {int(p.get('capacity_W', 0))} W"
                for i, p in enumerate(ports)
            }

            # Group all “car rows” by id (each row is one session in the flattened format)
            sessions_by_id = defaultdict(list)
            cap_by_id = {}

            for row in cars:
                cid = str(row.get("id", "") or f"Vehicle{len(sessions_by_id) + 1}")

                # capacity may be stored in different keys; normalize to float
                cap_val = row.get("battery_capacity", row.get("battery_capacity_wh", 0.0))
                try:
                    cap_by_id[cid] = float(cap_val)
                except Exception:
                    cap_by_id[cid] = 0.0

                # normalize times/dates
                def _date(v):
                    try:
                        return _dt.date.fromisoformat(str(v))
                    except Exception:
                        return _dt.date.today()

                def _time(v):
                    try:
                        return _dt.time.fromisoformat(str(v))
                    except Exception:
                        return _dt.time(0, 0)

                ad = _date(row.get("arrival_date"))
                at = _time(row.get("arrival_time"))
                dd = _date(row.get("departure_date"))
                dt = _time(row.get("departure_time"))
                port_id = int(row.get("charging_port", row.get("port_id", 1)))

                sessions_by_id[cid].append({
                    "arr_date": ad, "arr_time": at,
                    "dep_date": dd, "dep_time": dt,
                    "port_id": port_id,
                })

            # Now populate session_state as real multi-session vehicles
            st.session_state["num_ev"] = len(sessions_by_id)
            for i, (cid, sess_list) in enumerate(sessions_by_id.items()):
                st.session_state[f"carid_{i}"] = cid
                st.session_state[f"bcap_{i}"] = cap_by_id.get(cid, 0.0)
                st.session_state[f"num_sessions_{i}"] = len(sess_list)

                for j, s in enumerate(sess_list):
                    st.session_state[f"arr_date_{i}_{j}"] = s["arr_date"]
                    st.session_state[f"arr_time_{i}_{j}"] = s["arr_time"]
                    st.session_state[f"dep_date_{i}_{j}"] = s["dep_date"]
                    st.session_state[f"dep_time_{i}_{j}"] = s["dep_time"]
                    st.session_state[f"port_id_{i}_{j}"] = s["port_id"]
                    st.session_state[f"port_label_{i}_{j}"] = port_label.get(
                        s["port_id"], f"Port {s['port_id']}"
                    )

            loaded_configs.append(
                f"EV Configuration ({len(sessions_by_id)} vehicles, {len(cars)} total sessions) and Charging Ports"
            )

    # Create charging ports from EV configurations
    if ev_config and "InitializationSettings" in ev_config:
        cars = ev_config["InitializationSettings"].get("cars", [])
        if cars:
            # Extract unique charging port capacities
            port_capacities = set()
            for car in cars:
                port_capacity = car.get("charging_port", 0.0)
                if port_capacity > 0:
                    port_capacities.add(port_capacity)

            # Create ports list
            if port_capacities:
                ports = [{"capacity_W": float(capacity), "port_id": idx + 1} for idx, capacity in
                         enumerate(sorted(port_capacities))]
                st.session_state["ports"] = ports

            loaded_configs.append(f"EV Configuration ({len(cars)} vehicles) and Charging Ports")

    # Parse solar config
    if solar_config and "solar_panels" in solar_config:
        solar_panels = solar_config["solar_panels"]
        if solar_panels:
            st.session_state["num_solar_types"] = len(solar_panels)

            for i, panel in enumerate(solar_panels):
                st.session_state[f"lat_{i}"] = panel.get("latitude", 0.0)
                st.session_state[f"lon_{i}"] = panel.get("longitude", 0.0)
                st.session_state[f"alt_{i}"] = panel.get("altitude", 0)
                st.session_state[f"tilt_{i}"] = panel.get("surface_tilt", 0)
                st.session_state[f"panels_{i}"] = panel.get("number_of_panels", 1)
                st.session_state[f"module_id_{i}"] = panel.get("module_id", "")
                st.session_state[f"inverter_id_{i}"] = panel.get("inverter_id", "")

            loaded_configs.append(f"Solar Configuration ({len(solar_panels)} solar farms)")

    # Parse wind config
    if turbine_config and "wind_turbines" in turbine_config:
        wind_turbines = turbine_config["wind_turbines"]
        if wind_turbines:
            st.session_state["num_turbine_types"] = len(wind_turbines)

            for i, turbine in enumerate(wind_turbines):
                st.session_state[f"hub_{i}"] = turbine.get("hub_height", 0.0)
                st.session_state[f"count_{i}"] = turbine.get("number_of_turbines", 1)
                st.session_state[f"turbine_id_{i}"] = turbine.get("turbine_id", "")

            loaded_configs.append(f"Wind Configuration ({len(wind_turbines)} wind farms)")

    # Parse ports config (NEW)
    if ports_config:
        # Accept either {"ports":[...]} or {"InitializationSettings":..., "ports":[...]}
        ports = ports_config.get("ports", [])
        # If items don't include port_id, generate them
        normalized_ports = []
        for idx, p in enumerate(ports):
            # support either dicts with "capacity_W" (preferred) or "capacity_kW"
            cap_w = p.get("capacity_W", None)
            if cap_w is None and "capacity_kW" in p:
                try:
                    cap_w = float(p["capacity_kW"]) * 1000.0
                except Exception:
                    cap_w = 0.0
            try:
                cap_w = float(cap_w) if cap_w is not None else 0.0
            except Exception:
                cap_w = 0.0
            port_id = int(p.get("port_id", idx + 1))
            normalized_ports.append({"port_id": port_id, "capacity_W": cap_w})
        if normalized_ports:
            st.session_state["ports"] = normalized_ports
            loaded_configs.append(f"Charging Ports ({len(normalized_ports)} ports)")

    # Set default values for missing fields
    set_default_values_for_missing_fields()

    return loaded_configs


def parse_port_capacity_csv(csv_file):
    """
    Parse port capacity CSV file to get port IDs and their capacities.
    """
    try:
        import pandas as pd

        # Read CSV file
        df = pd.read_csv(csv_file)

        # Get unique charger IDs and their max power
        port_data = df.groupby('Ch ID')['Max power [kW]'].first().reset_index()

        ports = []
        for _, row in port_data.iterrows():
            port_id = int(row['Ch ID'])
            capacity_kw = float(row['Max power [kW]'])
            capacity_w = capacity_kw * 1000  # Convert to Watts

            ports.append({
                "port_id": port_id,
                "capacity_W": capacity_w,
                "capacity_kW": capacity_kw
            })

        # Sort by port ID
        ports.sort(key=lambda x: x["port_id"])

        return ports

    except Exception as e:
        st.error(f"Error parsing port capacity CSV: {e}")
        return []


def parse_car_csv_to_session_state(csv_file, port_capacity_file=None):
    """
    Parse uploaded car CSV file and populate session state with vehicle and session data.
    The CSV contains multiple sessions per vehicle with detailed charging information.
    """
    try:
        import pandas as pd
        from datetime import datetime

        # Read CSV file
        df = pd.read_csv(csv_file)

        # Parse port capacity file if provided
        ports = []
        if port_capacity_file:
            ports = parse_port_capacity_csv(port_capacity_file)

        # If no port capacity file, create default ports from charger numbers
        if not ports:
            unique_chargers = df['Charger #'].unique()
            ports = [{"capacity_W": 10000.0, "port_id": int(charger)} for charger in sorted(unique_chargers)]

        # Store ports in session state
        st.session_state["ports"] = ports

        # Create a mapping from port_id to port info for quick lookup
        port_lookup = {port["port_id"]: port for port in ports}

        # Group by Vehicle ID to get unique vehicles
        unique_vehicles = df['Vehicle ID'].unique()

        # Set number of vehicles
        st.session_state["num_ev"] = len(unique_vehicles)

        # Process each vehicle
        for vehicle_idx, vehicle_id in enumerate(unique_vehicles):
            vehicle_data = df[df['Vehicle ID'] == vehicle_id]

            # Get vehicle info (same for all sessions of this vehicle)
            first_session = vehicle_data.iloc[0]
            battery_capacity = first_session['Batt Cap [kWh]'] * 1000  # Convert to Wh

            # Set vehicle-level data
            st.session_state[f"carid_{vehicle_idx}"] = vehicle_id
            st.session_state[f"bcap_{vehicle_idx}"] = battery_capacity

            # Set number of sessions for this vehicle
            num_sessions = len(vehicle_data)
            st.session_state[f"num_sessions_{vehicle_idx}"] = num_sessions

            # Process each session for this vehicle
            for session_idx, (_, session) in enumerate(vehicle_data.iterrows()):
                # Parse arrival time (ETA)
                eta_str = session['Charging start time [ISO8601]']
                eta_datetime = pd.to_datetime(eta_str)
                st.session_state[f"arr_date_{vehicle_idx}_{session_idx}"] = eta_datetime.date()
                st.session_state[f"arr_time_{vehicle_idx}_{session_idx}"] = eta_datetime.time()

                # Parse departure time (ETD)
                etd_str = session['Charging end time [ISO8601]']
                etd_datetime = pd.to_datetime(etd_str)
                st.session_state[f"dep_date_{vehicle_idx}_{session_idx}"] = etd_datetime.date()
                st.session_state[f"dep_time_{vehicle_idx}_{session_idx}"] = etd_datetime.time()

                # Get charging port (Charger #)
                charger_num = int(session['Charger #'])
                st.session_state[f"port_id_{vehicle_idx}_{session_idx}"] = charger_num

                # Set port label with actual capacity if available
                if charger_num in port_lookup:
                    port_info = port_lookup[charger_num]
                    capacity_w = port_info["capacity_W"]
                    st.session_state[
                        f"port_label_{vehicle_idx}_{session_idx}"] = f"Port {charger_num} — {int(capacity_w)} W"
                else:
                    st.session_state[f"port_label_{vehicle_idx}_{session_idx}"] = f"Port {charger_num}"

        return f"Loaded {len(unique_vehicles)} vehicles with {len(df)} total sessions and {len(ports)} charging ports"

    except Exception as e:
        st.error(f"Error parsing car CSV: {e}")
        return None


def set_default_values_for_missing_fields():
    """Set default values for any missing fields to ensure the UI works properly."""

    # Default system settings
    if "money" not in st.session_state:
        st.session_state["money"] = 0.0
    if "price_high" not in st.session_state:
        st.session_state["price_high"] = 0.0
    if "price_low" not in st.session_state:
        st.session_state["price_low"] = 0.0
    if "timestep" not in st.session_state:
        st.session_state["timestep"] = 0
    if "config_name" not in st.session_state:
        st.session_state["config_name"] = ""

    # Default storage and grid
    if "storage" not in st.session_state:
        st.session_state["storage"] = 0.0
    if "grid" not in st.session_state:
        st.session_state["grid"] = 0.0

    # Default load settings
    if "load" not in st.session_state:
        st.session_state["load"] = 0.0

    # Default EV settings
    if "num_ev" not in st.session_state:
        st.session_state["num_ev"] = 1

    # Default plot settings
    if "plot_port" not in st.session_state:
        st.session_state["plot_port"] = 1
    if "plot_car_id" not in st.session_state:
        st.session_state["plot_car_id"] = ""

    # Default solar settings
    if "num_solar_types" not in st.session_state:
        st.session_state["num_solar_types"] = 1

    # Default wind settings
    if "num_turbine_types" not in st.session_state:
        st.session_state["num_turbine_types"] = 1

    # Default charging ports
    if "ports" not in st.session_state:
        st.session_state["ports"] = []

    # Load profile defaults
    if "load_mode" not in st.session_state:
        st.session_state["load_mode"] = "constant"  # or "timeseries"
    if "load_timeseries" not in st.session_state:
        st.session_state["load_timeseries"] = []  # list[{"time": "...", "load_w": float}]

    # Ensure at least one EV has default values
    if st.session_state["num_ev"] > 0:
        for i in range(st.session_state["num_ev"]):
            if f"carid_{i}" not in st.session_state:
                st.session_state[f"carid_{i}"] = f"Vehicle{i + 1}"
            if f"bcap_{i}" not in st.session_state:
                st.session_state[f"bcap_{i}"] = 0.0
            if f"num_sessions_{i}" not in st.session_state:
                st.session_state[f"num_sessions_{i}"] = 1

            # Ensure at least one session has default values
            num_sessions = st.session_state.get(f"num_sessions_{i}", 1)
            for session_idx in range(num_sessions):
                if f"arr_date_{i}_{session_idx}" not in st.session_state:
                    st.session_state[f"arr_date_{i}_{session_idx}"] = datetime.date.today()
                if f"arr_time_{i}_{session_idx}" not in st.session_state:
                    st.session_state[f"arr_time_{i}_{session_idx}"] = datetime.time(0, 0)
                if f"dep_date_{i}_{session_idx}" not in st.session_state:
                    st.session_state[f"dep_date_{i}_{session_idx}"] = datetime.date.today()
                if f"dep_time_{i}_{session_idx}" not in st.session_state:
                    st.session_state[f"dep_time_{i}_{session_idx}"] = datetime.time(0, 0)
                if f"port_label_{i}_{session_idx}" not in st.session_state:
                    st.session_state[f"port_label_{i}_{session_idx}"] = "— Select a port —"
                if f"port_id_{i}_{session_idx}" not in st.session_state:
                    st.session_state[f"port_id_{i}_{session_idx}"] = 1

    # Ensure at least one solar panel has default values
    if st.session_state["num_solar_types"] > 0:
        for i in range(st.session_state["num_solar_types"]):
            if f"lat_{i}" not in st.session_state:
                st.session_state[f"lat_{i}"] = 0.0
            if f"lon_{i}" not in st.session_state:
                st.session_state[f"lon_{i}"] = 0.0
            if f"alt_{i}" not in st.session_state:
                st.session_state[f"alt_{i}"] = 0
            if f"tilt_{i}" not in st.session_state:
                st.session_state[f"tilt_{i}"] = 0
            if f"panels_{i}" not in st.session_state:
                st.session_state[f"panels_{i}"] = 1
            if f"module_id_{i}" not in st.session_state:
                st.session_state[f"module_id_{i}"] = ""
            if f"inverter_id_{i}" not in st.session_state:
                st.session_state[f"inverter_id_{i}"] = ""

    # Ensure at least one wind turbine has default values
    if st.session_state["num_turbine_types"] > 0:
        for i in range(st.session_state["num_turbine_types"]):
            if f"hub_{i}" not in st.session_state:
                st.session_state[f"hub_{i}"] = 0.0
            if f"count_{i}" not in st.session_state:
                st.session_state[f"count_{i}"] = 1
            if f"turbine_id_{i}" not in st.session_state:
                st.session_state[f"turbine_id_{i}"] = ""


# ----------------------------- SETTINGS TAB -----------------------------

def settings_tab():
    """Tab for system configuration settings"""
    st.header("⚙️ System Configuration")

    st.info(
        "💡 **Configure system settings and save complete configurations. Use 'Save Full Configuration' to save all data from all tabs to a single YAML file.**")

    col1, col2 = st.columns(2)

    with col1:
        initial_money = st.number_input(
            "Initial budget (€)",
            min_value=0.0,
            format="%.4f",
            value=st.session_state.get("money", 0.0),
            key="tab_money"
        )
        price_high = st.number_input(
            "Maximum price (€/kWh)",
            min_value=0.0,
            format="%.4f",
            value=st.session_state.get("price_high", 0.0),
            key="tab_price_high"
        )
        price_low = st.number_input(
            "Minimum price (€/kWh)",
            min_value=0.0,
            format="%.4f",
            value=st.session_state.get("price_low", 0.0),
            key="tab_price_low"
        )

    with col2:
        timestep = st.number_input(
            "Time step in the simulation (minutes)",
            min_value=0,
            format="%d",
            value=st.session_state.get("timestep", 0),
            key="tab_timestep"
        )
        config_name = st.text_input(
            "Configuration version name",
            value=st.session_state.get("config_name", ""),
            key="tab_config_name"
        )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save Settings", type="primary", key="tab_save_settings"):
            st.session_state["money"] = initial_money
            st.session_state["price_high"] = price_high
            st.session_state["price_low"] = price_low
            st.session_state["timestep"] = timestep
            st.session_state["config_name"] = config_name
            st.success("✅ Settings saved!")
            st.rerun()

    with col2:
        if st.button("💾 Save Full Configuration", type="secondary", key="tab_save_full_config"):
            if not config_name.strip():
                st.error("Please enter a configuration name before saving.")
            else:
                filepath, config_data = save_configuration_to_yaml(config_name)
                if filepath:
                    st.success(f"✅ Full configuration '{config_name}' saved successfully!")
                    st.info(f"📁 Saved to: {filepath}")
                    st.rerun()
                else:
                    st.error("❌ Failed to save configuration.")


# ----------------------------- BATTERY TAB ------------------------------

def battery_tab():
    """Tab for battery configuration"""
    st.header("🔋 Battery Configuration")

    storage_capacity = st.number_input(
        "Battery storage capacity (Wh)",
        min_value=0.0,
        value=st.session_state.get("storage", 0.0),
        key="tab_storage"
    )

    if st.button("Save Battery Settings", type="primary", key="tab_save_battery"):
        st.session_state["storage"] = storage_capacity
        st.success("Battery settings saved!")
        st.rerun()


# ----------------------------- GRID TAB --------------------------------

def grid_tab():
    """Tab for grid configuration"""
    st.header("⚡ Electrical Grid Configuration")

    grid_capacity = st.number_input(
        "Grid capacity (W)",
        min_value=0.0,
        value=st.session_state.get("grid", 0.0),
        key="tab_grid"
    )

    if st.button("Save Grid Settings", type="primary", key="tab_save_grid"):
        st.session_state["grid"] = grid_capacity
        st.success("Grid settings saved!")
        st.rerun()


# ----------------------------- LOAD TAB --------------------------------

def load_tab():
    """Tab for additional load configuration"""
    st.header("🏠 Load Configuration")

    mode = st.radio(
        "Load mode",
        options=["constant", "timeseries"],
        index=0 if st.session_state.get("load_mode","constant") == "constant" else 1,
        key="tab_load_mode",
        help="Constant applies one value at all times; Timeseries reads points from CSV/YAML."
    )

    if mode == "constant":
        add_load = st.number_input(
            "Constant Load (W)",
            min_value=0.0,
            value=float(st.session_state.get("load", 0.0)),
            key="tab_load_const",
        )
        if st.button("💾 Save Load Settings", type="primary", key="tab_save_load_manual"):
            st.session_state["load_mode"] = "constant"
            st.session_state["load"] = float(add_load)
            st.session_state["load_timeseries"] = []
            st.success("✅ Constant load saved!")
            st.rerun()

    else:
        st.caption("Edit or inspect your timeseries below (ISO8601 time + load in Watts).")
        # Prepare editable dataframe
        import pandas as pd
        rows = st.session_state.get("load_timeseries", [])
        df = pd.DataFrame(rows if rows else [{"time": "", "load_w": 0.0}])
        edited = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            key="tab_load_editor",
        )
        if st.button("💾 Save Timeseries", type="primary", key="tab_save_load_ts"):
            # normalize
            records = []
            for _, r in edited.iterrows():
                if str(r.get("time","")).strip() == "":
                    continue
                records.append({"time": str(r["time"]), "load_w": float(r.get("load_w", 0.0) or 0.0)})
            st.session_state["load_mode"] = "timeseries"
            st.session_state["load_timeseries"] = records
            st.success(f"✅ Saved {len(records)} load points!")
            st.rerun()


# ----------------------------- CHARGING PORTS TAB -----------------------

def charging_ports_tab():
    """Tab for charging ports configuration"""
    st.header("⚡ Charging Port Configuration")

    num_ports = st.number_input(
        "Number of charging ports",
        min_value=1,
        step=1,
        value=len(st.session_state.get("ports", [])) or 1,
        key="tab_cp_num_ports",
    )

    # inputs for each port
    for i in range(int(num_ports)):
        st.subheader(f"Port {i + 1}")
        st.number_input(
            f"Port {i + 1} capacity (W)",
            min_value=0.0,
            value=(
                float(st.session_state.get("ports", [{}] * int(num_ports))[i].get("capacity_W", 0.0))
                if i < len(st.session_state.get("ports", []))
                else 0.0
            ),
            key=f"tab_cp_port_cap_{i}",
        )

    st.markdown("---")
    # Choose which port to plot (1-based)
    port_max = int(num_ports) if int(num_ports) > 0 else 1
    plot_port = st.number_input(
        "Port number to plot (1-based)",
        min_value=1,
        max_value=port_max,
        value=int(st.session_state.get("plot_port", 1)) if st.session_state.get("plot_port") else 1,
        step=1,
        key="tab_cp_plot_port",
    )

    # save button
    if st.button("Save Ports", type="primary", key="tab_cp_save_btn"):
        ports = []
        for i in range(int(num_ports)):
            ports.append({"capacity_W": float(st.session_state[f"tab_cp_port_cap_{i}"]), "port_id": i + 1})
        st.session_state["ports"] = ports
        # persist selection
        st.session_state["plot_port"] = int(plot_port)
        st.success(f"Saved {len(ports)} charging port(s). Plotting Port {plot_port}.")
        st.rerun()


# ----------------------------- EVs TAB ---------------------------------

def evs_tab():
    """Tab for EV configuration with multiple sessions per car"""
    st.header("🚗 Electric Vehicle Configuration")

    st.info(
        "💡 **Tip**: Upload EV data files in the **📁 Files** tab to automatically populate this configuration, or enter data manually below.")

    # Get number of vehicles
    num_ev = st.number_input(
        "Number of EVs",
        min_value=1,
        step=1,
        value=st.session_state.get("num_ev", 1),
        key="tab_num_ev",
    )

    # Build selectable list of charging ports
    ports = st.session_state.get("ports", [])
    port_options = ["— Select a port —"]
    for idx, port in enumerate(ports):
        port_id = port.get("port_id", idx + 1)
        capacity = port.get("capacity_W", 10000)
        port_options.append(f"Port {port_id} — {int(capacity)} W")

    # Inputs for each EV
    for i in range(int(num_ev)):
        st.subheader(f"🚗 Vehicle {i + 1}")

        # Vehicle-level information
        col1, col2 = st.columns(2)
        with col1:
            st.text_input(
                f"Vehicle ID",
                value=st.session_state.get(f"carid_{i}", f"Vehicle{i + 1}"),
                key=f"tab_carid_{i}",
                help="Unique identifier for this vehicle"
            )
        with col2:
            st.number_input(
                f"Battery Capacity (Wh)",
                min_value=0.0,
                value=st.session_state.get(f"bcap_{i}", 0.0),
                key=f"tab_bcap_{i}",
                help="Total battery capacity in Watt-hours"
            )

        # Sessions for this vehicle
        num_sessions = st.number_input(
            f"Number of charging sessions for Vehicle {i + 1}",
            min_value=1,
            step=1,
            value=st.session_state.get(f"num_sessions_{i}", 1),
            key=f"tab_num_sessions_{i}",
        )

        # Session inputs
        for session_idx in range(int(num_sessions)):
            with st.expander(f"📅 Session {session_idx + 1} - Charging Details", expanded=session_idx == 0):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.date_input(
                        f"Arrival Date",
                        value=st.session_state.get(f"arr_date_{i}_{session_idx}", datetime.date.today()),
                        key=f"tab_arr_date_{i}_{session_idx}",
                    )
                    st.time_input(
                        f"Arrival Time",
                        value=st.session_state.get(f"arr_time_{i}_{session_idx}", datetime.time(0, 0)),
                        key=f"tab_arr_time_{i}_{session_idx}",
                    )

                with col2:
                    st.date_input(
                        f"Departure Date",
                        value=st.session_state.get(f"dep_date_{i}_{session_idx}", datetime.date.today()),
                        key=f"tab_dep_date_{i}_{session_idx}",
                    )
                    st.time_input(
                        f"Departure Time",
                        value=st.session_state.get(f"dep_time_{i}_{session_idx}", datetime.time(0, 0)),
                        key=f"tab_dep_time_{i}_{session_idx}",
                    )

                with col3:
                    # Port selection
                    current_port = st.session_state.get(f"port_label_{i}_{session_idx}", "— Select a port —")
                    port_index = 0
                    if current_port in port_options:
                        port_index = port_options.index(current_port)

                    selected_port = st.selectbox(
                        f"Charging Port",
                        port_options,
                        index=port_index,
                        key=f"tab_port_{i}_{session_idx}",
                        help="Select the charging port for this session"
                    )

                    # Store the selected port ID for this session
                    if selected_port != "— Select a port —":
                        # Extract port ID from the selected option
                        try:
                            port_id = int(selected_port.split("Port ")[1].split(" —")[0])
                            st.session_state[f"port_id_{i}_{session_idx}"] = port_id
                        except:
                            st.session_state[f"port_id_{i}_{session_idx}"] = 1

        st.markdown("---")

    # Live "Car to plot" selector from currently visible IDs
    car_ids_live = [st.session_state.get(f"tab_carid_{i}", f"Vehicle{i + 1}") for i in range(int(num_ev))]
    if car_ids_live:
        default_idx = 0
        if st.session_state.get("plot_car_id") in car_ids_live:
            default_idx = car_ids_live.index(st.session_state["plot_car_id"])
        chosen = st.selectbox("Car to plot", car_ids_live, index=default_idx, key="tab_ev_plot_car_select")
        st.session_state["plot_car_id"] = chosen

    # Save configuration
    if st.button("💾 Save EV Configuration", type="primary", key="tab_save_ev_manual"):
        try:
            # Validate and save all data
            for i in range(int(num_ev)):
                # Save vehicle-level data
                st.session_state[f"carid_{i}"] = st.session_state.get(f"tab_carid_{i}", f"Vehicle{i + 1}")
                st.session_state[f"bcap_{i}"] = st.session_state.get(f"tab_bcap_{i}", 0.0)

                # Save session data
                num_sessions = st.session_state.get(f"tab_num_sessions_{i}", 1)
                st.session_state[f"num_sessions_{i}"] = int(num_sessions)

                for session_idx in range(int(num_sessions)):
                    st.session_state[f"arr_date_{i}_{session_idx}"] = st.session_state.get(
                        f"tab_arr_date_{i}_{session_idx}", datetime.date.today())
                    st.session_state[f"arr_time_{i}_{session_idx}"] = st.session_state.get(
                        f"tab_arr_time_{i}_{session_idx}", datetime.time(0, 0))
                    st.session_state[f"dep_date_{i}_{session_idx}"] = st.session_state.get(
                        f"tab_dep_date_{i}_{session_idx}", datetime.date.today())
                    st.session_state[f"dep_time_{i}_{session_idx}"] = st.session_state.get(
                        f"tab_dep_time_{i}_{session_idx}", datetime.time(0, 0))
                    st.session_state[f"port_label_{i}_{session_idx}"] = st.session_state.get(
                        f"tab_port_{i}_{session_idx}", "— Select a port —")

                    # Save port ID if available
                    if f"port_id_{i}_{session_idx}" in st.session_state:
                        st.session_state[f"port_id_{i}_{session_idx}"] = st.session_state[f"port_id_{i}_{session_idx}"]

            st.session_state["num_ev"] = int(num_ev)
            st.success("✅ EV configuration saved successfully!")

        except Exception as e:
            st.error(f"❌ Error saving EV configuration: {e}")

    # Display summary
    if st.session_state.get("num_ev", 0) > 0:
        st.subheader("📊 Configuration Summary")
        total_sessions = sum(
            st.session_state.get(f"num_sessions_{i}", 0) for i in range(st.session_state.get("num_ev", 0)))
        st.info(
            f"**{st.session_state.get('num_ev', 0)} vehicles** with **{total_sessions} total charging sessions** configured.")


# ----------------------------- WIND TAB --------------------------------

def wind_tab():
    """Tab for wind turbine configuration"""
    st.header("💨 Wind Turbine Configuration")

    num_turbine_types = st.number_input(
        "Number of wind farms",
        min_value=1,
        step=1,
        value=st.session_state.get("num_turbine_types", 1),
        key="tab_num_turbine_types"
    )

    turbines = []
    for i in range(int(num_turbine_types)):
        st.subheader(f"Turbine farm {i + 1}")

        col1, col2, col3 = st.columns(3)

        with col1:
            hub_height = st.number_input(
                f"Hub height (m) for a turbine in farm {i + 1}",
                min_value=0.0,
                value=st.session_state.get(f"hub_{i}", 0.0),
                key=f"tab_hub_{i}"
            )

        with col2:
            number_of_turbines = st.number_input(
                f"Number of turbines in farm {i + 1}",
                min_value=1,
                step=1,
                value=st.session_state.get(f"count_{i}", 1),
                key=f"tab_count_{i}"
            )

        with col3:
            turbine_id = st.text_input(
                f"Turbine name (e.g., E-126/4200) for farm {i + 1}",
                value=st.session_state.get(f"turbine_id_{i}", ""),
                key=f"tab_turbine_id_{i}"
            )

        turbines.append({
            "hub_height": hub_height,
            "number_of_turbines": number_of_turbines,
            "turbine_id": turbine_id,
        })

    if st.button("Save Wind Configuration", type="primary", key="tab_save_wind"):
        st.session_state["num_turbine_types"] = num_turbine_types
        for i in range(int(num_turbine_types)):
            st.session_state[f"hub_{i}"] = st.session_state[f"tab_hub_{i}"]
            st.session_state[f"count_{i}"] = st.session_state[f"tab_count_{i}"]
            st.session_state[f"turbine_id_{i}"] = st.session_state[f"tab_turbine_id_{i}"]
        st.success("Wind turbine configuration saved!")
        st.rerun()


# ----------------------------- SOLAR TAB -------------------------------

def solar_tab():
    """Tab for solar panel configuration"""
    st.header("☀️ Solar Panel Configuration")

    num_solar_types = st.number_input(
        "Number of solar farms",
        min_value=1,
        step=1,
        value=st.session_state.get("num_solar_types", 1),
        key="tab_num_solar_types"
    )

    solar_panels = []
    for i in range(int(num_solar_types)):
        st.subheader(f"Solar farm {i + 1}")

        col1, col2 = st.columns(2)

        with col1:
            latitude = st.number_input(
                f"Latitude (°) for panels in farm {i + 1}",
                format="%.4f",
                min_value=-180.0,
                max_value=180.0,
                value=st.session_state.get(f"lat_{i}", 0.0),
                key=f"tab_lat_{i}"
            )
            longitude = st.number_input(
                f"Longitude (°) for panels in farm {i + 1}",
                format="%.4f",
                min_value=-180.0,
                max_value=180.0,
                value=st.session_state.get(f"lon_{i}", 0.0),
                key=f"tab_lon_{i}"
            )
            altitude = st.number_input(
                f"Altitude (m) for panels in farm {i + 1}",
                value=st.session_state.get(f"alt_{i}", 0),
                key=f"tab_alt_{i}"
            )

        with col2:
            surface_tilt = st.number_input(
                f"Surface tilt (°) for panels in farm {i + 1}",
                value=st.session_state.get(f"tilt_{i}", 0),
                key=f"tab_tilt_{i}"
            )
            number_of_panels = st.number_input(
                f"Number of panels in farm {i + 1}",
                min_value=1,
                step=1,
                value=st.session_state.get(f"panels_{i}", 1),
                key=f"tab_panels_{i}"
            )
            module_id = st.text_input(
                f"Panel name for farm {i + 1}",
                value=st.session_state.get(f"module_id_{i}", ""),
                key=f"tab_module_id_{i}"
            )
            inverter_id = st.text_input(
                f"Inverter name for farm {i + 1}",
                value=st.session_state.get(f"inverter_id_{i}", ""),
                key=f"tab_inverter_id_{i}"
            )

        solar_panels.append({
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
            "surface_tilt": surface_tilt,
            "number_of_panels": number_of_panels,
            "module_id": module_id,
            "inverter_id": inverter_id,
        })

    if st.button("Save Solar Configuration", type="primary", key="tab_save_solar"):
        st.session_state["num_solar_types"] = num_solar_types
        for i in range(int(num_solar_types)):
            st.session_state[f"lat_{i}"] = st.session_state[f"tab_lat_{i}"]
            st.session_state[f"lon_{i}"] = st.session_state[f"tab_lon_{i}"]
            st.session_state[f"alt_{i}"] = st.session_state[f"tab_alt_{i}"]
            st.session_state[f"tilt_{i}"] = st.session_state[f"tab_tilt_{i}"]
            st.session_state[f"panels_{i}"] = st.session_state[f"tab_panels_{i}"]
            st.session_state[f"inverter_id_{i}"] = st.session_state[f"tab_inverter_id_{i}"]
            st.session_state[f"module_id_{i}"] = st.session_state[f"tab_module_id_{i}"]

        st.success("Solar panel configuration saved!")
        st.rerun()


# ----------------------------- FILE UPLOAD TAB -------------------------

def file_upload_tab():
    """Tab for file uploads that populate session storage"""
    st.header("📁 File Uploads")

    st.info(
        "💡 **Upload configuration files to automatically populate all tab fields. The uploaded data will be stored in session storage and used throughout the application.**")

    # YAML Upload Section
    st.subheader("📂 YAML Configuration Files")
    st.write(
        "Upload YAML configuration files to populate system settings, battery, grid, load, solar, wind, and charging port configurations:")

    col1, col2 = st.columns(2)

    with col1:
        up_controller = st.file_uploader("System Settings (controller_add_config.yaml)", type=["yaml", "yml"],
                                         key="tab_up_controller_yaml")
        up_ev = st.file_uploader("EV Configuration (ev_config.yaml)", type=["yaml", "yml"], key="tab_up_ev_yaml")
        # NEW: Ports YAML uploader
        up_ports_yaml = st.file_uploader("Charging Ports (port_data.yaml)", type=["yaml", "yml"],
                                         key="tab_up_ports_yaml")

    with col2:
        up_solar = st.file_uploader("Solar Configuration (solar_config.yaml)", type=["yaml", "yml"],
                                    key="tab_up_solar_yaml")
        up_turbine = st.file_uploader("Wind Configuration (turbine_config.yaml)", type=["yaml", "yml"],
                                      key="tab_up_turbine_yaml")
        up_load_yaml = st.file_uploader("Load Profile (load_config.yaml)", type=["yaml", "yml"], key="tab_up_load_yaml")

    def _validate_yaml(file_obj, required_top_keys=()):
        data = yaml.safe_load(file_obj.getvalue().decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("YAML root must be a mapping (dictionary).")
        for k in required_top_keys:
            if k not in data:
                raise ValueError(f"Missing top-level key: '{k}'")
        return data

    # Allow partial uploads - upload only the files that are provided
    uploaded_files = []
    if up_controller:
        uploaded_files.append(("controller_add_config.yaml", up_controller))
    if up_ev:
        uploaded_files.append(("ev_config.yaml", up_ev))
    if up_solar:
        uploaded_files.append(("solar_config.yaml", up_solar))
    if up_turbine:
        uploaded_files.append(("turbine_config.yaml", up_turbine))
    if up_ports_yaml:
        uploaded_files.append(("port_data.yaml", up_ports_yaml))
    if up_load_yaml:
        uploaded_files.append(("load_config.yaml", up_load_yaml))

    if uploaded_files:
        if st.button("📥 Load YAML Configurations", type="primary", key="tab_save_yamls"):
            try:
                os.makedirs("configurations", exist_ok=True)

                # Initialize config variables
                controller_config = None
                ev_config = None
                solar_config = None
                turbine_config = None
                ports_config = None
                load_config = None
                # Process each uploaded file
                for filename, file_obj in uploaded_files:
                    # Validate and parse YAML
                    config_data = _validate_yaml(file_obj, required_top_keys=("InitializationSettings",))

                    # Save file
                    with open(f"configurations/{filename}", "wb") as f:
                        f.write(file_obj.getbuffer())

                    # Store config for parsing
                    if filename == "controller_add_config.yaml":
                        controller_config = config_data
                    elif filename == "ev_config.yaml":
                        ev_config = config_data
                    elif filename == "solar_config.yaml":
                        solar_config = config_data
                    elif filename == "turbine_config.yaml":
                        turbine_config = config_data
                    elif filename == "port_data.yaml":
                        ports_config = config_data
                    elif filename == "load_config.yaml":
                        load_config = config_data

                # Parse YAML configs and populate session state
                loaded_configs = parse_yaml_configs_to_session_state(controller_config, ev_config, solar_config,
                                                                     turbine_config, ports_config,load_config)

                # Flag to run directly from uploaded YAMLs
                st.session_state["use_uploaded_yaml"] = True

                uploaded_names = [name for name, _ in uploaded_files]
                if loaded_configs:
                    st.success(f"✅ YAML configurations loaded! ({', '.join(uploaded_names)})")
                    st.info(f"📋 **Loaded configurations:** {', '.join(loaded_configs)}")
                    st.success(
                        "All tab fields have been populated with the uploaded configuration values. You can now edit them in the tabs or run the simulation directly.")
                else:
                    st.success(f"✅ YAML configurations saved! ({', '.join(uploaded_names)})")
                st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")
    else:
        st.info("Upload one or more YAML configuration files above to populate the tab fields.")

    # Load existing YAML files
    st.markdown("---")
    st.subheader("📂 Load Existing YAML Files")

    if os.path.exists("configurations"):
        existing_files = [f for f in os.listdir("configurations") if f.endswith(('.yaml', '.yml'))]
        if existing_files:
            st.write("Found existing YAML files in the configurations folder:")
            for file in existing_files:
                st.write(f"• {file}")

            if st.button("📂 Load Existing YAML Files", type="secondary", key="tab_load_existing_yamls"):
                try:
                    # Load existing files
                    controller_config = None
                    ev_config = None
                    solar_config = None
                    turbine_config = None
                    ports_config = None
                    load_config = None

                    for file in existing_files:
                        file_path = os.path.join("configurations", file)
                        with open(file_path, 'r') as f:
                            config_data = yaml.safe_load(f)

                        if file == "controller_add_config.yaml":
                            controller_config = config_data
                        elif file == "ev_config.yaml":
                            ev_config = config_data
                        elif file == "solar_config.yaml":
                            solar_config = config_data
                        elif file == "turbine_config.yaml":
                            turbine_config = config_data
                        elif file == "port_data.yaml":
                            ports_config = config_data
                        elif file == "load_config.yaml":
                            load_config = config_data

                    # Parse and populate session state
                    loaded_configs = parse_yaml_configs_to_session_state(controller_config, ev_config, solar_config,
                                                                         turbine_config, ports_config,load_config)

                    st.session_state["use_uploaded_yaml"] = True
                    if loaded_configs:
                        st.success("✅ Existing YAML files loaded!")
                        st.info(f"📋 **Loaded configurations:** {', '.join(loaded_configs)}")
                        st.success("All tab fields have been populated with the configuration values.")
                    else:
                        st.success("✅ Existing YAML files loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load existing YAML files: {e}")
        else:
            st.info("No existing YAML files found in the configurations folder.")
    else:
        st.info("No configurations folder found.")

    st.markdown("---")

    # EV Data Files Upload Section
    st.subheader("🚗 EV Data Files")
    st.write("Upload EV data files to populate vehicle and charging session information:")

    os.makedirs("data", exist_ok=True)

    col1, col2 = st.columns(2)

    with col1:
        up_cars = st.file_uploader("EV Cars Data (data_cars.csv)", type=["csv"], key="tab_up_ev_cars")

    with col2:
        up_ports = st.file_uploader("Port Capacity Data (port_capacity_car.csv)", type=["csv"], key="tab_up_ev_ports")

    if up_cars and up_ports:
        if st.button("🚗 Load EV Data", type="primary", key="tab_btn_save_ev_files"):
            try:
                # Save with expected names
                cars_path = os.path.join("data", "data_cars.csv")
                ports_path = os.path.join("data", "port_capacity_car.csv")

                with open(cars_path, "wb") as f:
                    f.write(up_cars.getbuffer())
                with open(ports_path, "wb") as f:
                    f.write(up_ports.getbuffer())

                st.session_state["cars_path"] = cars_path
                st.session_state["bcap_path"] = ports_path

                # Parse car CSV and populate session state
                result = parse_car_csv_to_session_state(up_cars, up_ports)
                if result:
                    st.success(f"✅ EV data loaded successfully!")
                    st.info(f"📋 **{result}** - All EV tab fields have been populated with the uploaded data.")
                else:
                    st.success(f"✅ EV data saved successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"EV data upload failed: {e}")
    elif up_cars or up_ports:
        st.warning("Please upload both EV cars data and port capacity data files.")
    else:
        st.info("Upload both EV data files above to populate the EV configuration.")
    st.subheader("🏠 Load Data Files")
    up_load_csv = st.file_uploader("Load profile (CSV)", type=["csv"], key="tab_up_load_csv")

    if up_load_csv:
        if st.button("📥 Load Load CSV", type="primary", key="tab_btn_load_csv"):
            try:
                msg = parse_load_csv_to_session_state(up_load_csv)
                if msg: st.success("✅ " + msg)
                st.rerun()
            except Exception as e:
                st.error(f"Load CSV upload failed: {e}")
    else:
        st.info("Upload a Load CSV with columns 'Time [ISO8601]' and 'Consumption without charging [kW]'.")


# ----------------------------- CONFIGURATION MANAGEMENT --------------------------

def collect_all_config_data():
    """
    Collect all configuration data from session state and organize it into a structured format.
    This includes data from all tabs except the Files tab.
    """
    config_data = {
        "metadata": {
            "config_name": st.session_state.get("config_name", ""),
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "1.0",
            "description": "Complete system configuration including all components and settings"
        },

        "system_settings": {
            "initial_budget": st.session_state.get("money", 0.0),
            "price_high": st.session_state.get("price_high", 0.0),
            "price_low": st.session_state.get("price_low", 0.0),
            "timestep_minutes": st.session_state.get("timestep", 0)
        },

        "battery_storage": {
            "storage_capacity_wh": st.session_state.get("storage", 0.0)
        },

        "grid_connection": {
            "grid_capacity_w": st.session_state.get("grid", 0.0)
        },


        "charging_ports": st.session_state.get("ports", []),

        "electric_vehicles": {
            "num_vehicles": st.session_state.get("num_ev", 0),
            "vehicles": []
        },

        "wind_turbines": {
            "num_turbine_types": st.session_state.get("num_turbine_types", 0),
            "turbines": []
        },

        "solar_panels": {
            "num_solar_types": st.session_state.get("num_solar_types", 0),
            "solar_panels": []
        },

        "plot_settings": {
            "plot_port": st.session_state.get("plot_port", 1),
            "plot_car_id": st.session_state.get("plot_car_id", "")
        }
    }

    if st.session_state.get("load_mode", "constant") == "constant":
        load_profile = {
            "mode": "constant",
            "constant_load_w": float(st.session_state.get("load", 0.0))
        }
    else:
        load_profile = {
            "mode": "timeseries",
            "points": st.session_state.get("load_timeseries", [])
        }

    config_data["load_profile"] = load_profile

    # EV Configuration (from EVs tab)
    num_ev = st.session_state.get("num_ev", 0)
    for i in range(num_ev):
        vehicle = {
            "vehicle_id": st.session_state.get(f"carid_{i}", f"Vehicle{i + 1}"),
            "battery_capacity_wh": st.session_state.get(f"bcap_{i}", 0.0),
            "num_sessions": st.session_state.get(f"num_sessions_{i}", 1),
            "charging_sessions": []
        }

        num_sessions = st.session_state.get(f"num_sessions_{i}", 1)
        for session_idx in range(num_sessions):
            session = {
                "arrival_date": st.session_state.get(f"arr_date_{i}_{session_idx}", datetime.date.today()).isoformat(),
                "arrival_time": st.session_state.get(f"arr_time_{i}_{session_idx}", datetime.time(0, 0)).isoformat(),
                "departure_date": st.session_state.get(f"dep_date_{i}_{session_idx}",
                                                       datetime.date.today()).isoformat(),
                "departure_time": st.session_state.get(f"dep_time_{i}_{session_idx}", datetime.time(0, 0)).isoformat(),
                "port_id": st.session_state.get(f"port_id_{i}_{session_idx}", 1),
                "port_label": st.session_state.get(f"port_label_{i}_{session_idx}", "— Select a port —")
            }
            vehicle["charging_sessions"].append(session)

        config_data["electric_vehicles"]["vehicles"].append(vehicle)

    # Wind Configuration (from Wind tab)
    num_turbine_types = st.session_state.get("num_turbine_types", 0)
    for i in range(num_turbine_types):
        turbine = {
            "hub_height_m": st.session_state.get(f"hub_{i}", 0.0),
            "number_of_turbines": st.session_state.get(f"count_{i}", 1),
            "turbine_id": st.session_state.get(f"turbine_id_{i}", "")
        }
        config_data["wind_turbines"]["turbines"].append(turbine)

    # Solar Configuration (from Solar tab)
    num_solar_types = st.session_state.get("num_solar_types", 0)
    for i in range(num_solar_types):
        panel = {
            "latitude": st.session_state.get(f"lat_{i}", 0.0),
            "longitude": st.session_state.get(f"lon_{i}", 0.0),
            "altitude_m": st.session_state.get(f"alt_{i}", 0),
            "surface_tilt_degrees": st.session_state.get(f"tilt_{i}", 0),
            "number_of_panels": st.session_state.get(f"panels_{i}", 1),
            "module_id": st.session_state.get(f"module_id_{i}", ""),
            "inverter_id": st.session_state.get(f"inverter_id_{i}", "")
        }
        config_data["solar_panels"]["solar_panels"].append(panel)

    return config_data


def save_configuration_to_yaml(config_name):
    """
    Save the current configuration to a YAML file.
    """
    try:
        # Create saved_configurations directory if it doesn't exist
        os.makedirs("saved_configurations", exist_ok=True)

        # Collect all configuration data
        config_data = collect_all_config_data()
        config_data["metadata"]["config_name"] = config_name

        # Create filename from config name (sanitize for filesystem)
        safe_name = "".join(c for c in config_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        filename = f"{safe_name}.yaml"
        filepath = os.path.join("saved_configurations", filename)

        # Save to YAML file with proper formatting
        with open(filepath, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, indent=2)

        return filepath, config_data

    except Exception as e:
        st.error(f"Error saving configuration: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None


def load_configuration_from_yaml(filepath):
    """
    Load configuration from a YAML file and populate session state.
    """
    try:
        with open(filepath, 'r') as f:
            config_data = yaml.safe_load(f)

        # Handle both old and new YAML formats for backward compatibility
        if "metadata" in config_data:
            # New format
            metadata = config_data["metadata"]
            st.session_state["config_name"] = metadata.get("config_name", "")
            # Load profile (new format)
            if "load_profile" in config_data:
                lp = config_data["load_profile"]
                st.session_state["load_mode"] = lp.get("mode", "constant")
                if st.session_state["load_mode"] == "constant":
                    st.session_state["load"] = float(lp.get("constant_load_w", 0.0))
                    st.session_state["load_timeseries"] = []
                else:
                    st.session_state["load_timeseries"] = lp.get("points", [])

            # Load system settings
            if "system_settings" in config_data:
                sys_settings = config_data["system_settings"]
                st.session_state["money"] = sys_settings.get("initial_budget", 0.0)
                st.session_state["price_high"] = sys_settings.get("price_high", 0.0)
                st.session_state["price_low"] = sys_settings.get("price_low", 0.0)
                st.session_state["timestep"] = sys_settings.get("timestep_minutes", 0)

            # Load battery settings
            if "battery_storage" in config_data:
                st.session_state["storage"] = config_data["battery_storage"].get("storage_capacity_wh", 0.0)

            # Load grid settings
            if "grid_connection" in config_data:
                st.session_state["grid"] = config_data["grid_connection"].get("grid_capacity_w", 0.0)

            # Load load settings
            if "load_profile" in config_data:
                st.session_state["load"] = config_data["load_profile"].get("constant_load_w", 0.0)

            # Load charging ports
            if "charging_ports" in config_data:
                st.session_state["ports"] = config_data["charging_ports"]

            # Load EV configuration
            if "electric_vehicles" in config_data:
                ev_config = config_data["electric_vehicles"]
                st.session_state["num_ev"] = ev_config.get("num_vehicles", 0)

                for i, vehicle in enumerate(ev_config.get("vehicles", [])):
                    st.session_state[f"carid_{i}"] = vehicle.get("vehicle_id", f"Vehicle{i + 1}")
                    st.session_state[f"bcap_{i}"] = vehicle.get("battery_capacity_wh", 0.0)
                    st.session_state[f"num_sessions_{i}"] = vehicle.get("num_sessions", 1)

                    for session_idx, session in enumerate(vehicle.get("charging_sessions", [])):
                        # Parse dates and times
                        st.session_state[f"arr_date_{i}_{session_idx}"] = datetime.datetime.fromisoformat(
                            session.get("arrival_date", datetime.date.today().isoformat())).date()
                        st.session_state[f"arr_time_{i}_{session_idx}"] = datetime.datetime.fromisoformat(
                            session.get("arrival_time", datetime.time(0, 0).isoformat())).time()
                        st.session_state[f"dep_date_{i}_{session_idx}"] = datetime.datetime.fromisoformat(
                            session.get("departure_date", datetime.date.today().isoformat())).date()
                        st.session_state[f"dep_time_{i}_{session_idx}"] = datetime.datetime.fromisoformat(
                            session.get("departure_time", datetime.time(0, 0).isoformat())).time()
                        st.session_state[f"port_id_{i}_{session_idx}"] = session.get("port_id", 1)
                        st.session_state[f"port_label_{i}_{session_idx}"] = session.get("port_label",
                                                                                        "— Select a port —")

            # Load wind configuration
            if "wind_turbines" in config_data:
                wind_config = config_data["wind_turbines"]
                st.session_state["num_turbine_types"] = wind_config.get("num_turbine_types", 0)

                for i, turbine in enumerate(wind_config.get("turbines", [])):
                    st.session_state[f"hub_{i}"] = turbine.get("hub_height_m", 0.0)
                    st.session_state[f"count_{i}"] = turbine.get("number_of_turbines", 1)
                    st.session_state[f"turbine_id_{i}"] = turbine.get("turbine_id", "")

            # Load solar configuration
            if "solar_panels" in config_data:
                solar_config = config_data["solar_panels"]
                st.session_state["num_solar_types"] = solar_config.get("num_solar_types", 0)

                for i, panel in enumerate(solar_config.get("solar_panels", [])):
                    st.session_state[f"lat_{i}"] = panel.get("latitude", 0.0)
                    st.session_state[f"lon_{i}"] = panel.get("longitude", 0.0)
                    st.session_state[f"alt_{i}"] = panel.get("altitude_m", 0)
                    st.session_state[f"tilt_{i}"] = panel.get("surface_tilt_degrees", 0)
                    st.session_state[f"panels_{i}"] = panel.get("number_of_panels", 1)
                    st.session_state[f"module_id_{i}"] = panel.get("module_id", "")
                    st.session_state[f"inverter_id_{i}"] = panel.get("inverter_id", "")

            # Load plot settings
            if "plot_settings" in config_data:
                plot_settings = config_data["plot_settings"]
                st.session_state["plot_port"] = plot_settings.get("plot_port", 1)
                st.session_state["plot_car_id"] = plot_settings.get("plot_car_id", "")

        else:
            # Legacy format
            st.session_state["config_name"] = config_data.get("config_name", "")

            # Load system settings
            if "system_settings" in config_data:
                sys_settings = config_data["system_settings"]
                st.session_state["money"] = sys_settings.get("money", 0.0)
                st.session_state["price_high"] = sys_settings.get("price_high", 0.0)
                st.session_state["price_low"] = sys_settings.get("price_low", 0.0)
                st.session_state["timestep"] = sys_settings.get("timestep", 0)

            # Load battery settings
            if "battery_settings" in config_data:
                st.session_state["storage"] = config_data["battery_settings"].get("storage_capacity", 0.0)

            # Load grid settings
            if "grid_settings" in config_data:
                st.session_state["grid"] = config_data["grid_settings"].get("grid_capacity", 0.0)

            # Load load settings
            if "load_settings" in config_data:
                st.session_state["load"] = config_data["load_settings"].get("constant_load", 0.0)

            # Load charging ports
            if "charging_ports" in config_data:
                st.session_state["ports"] = config_data["charging_ports"]

            # Load EV configuration
            if "ev_configuration" in config_data:
                ev_config = config_data["ev_configuration"]
                st.session_state["num_ev"] = ev_config.get("num_vehicles", 0)

                for i, vehicle in enumerate(ev_config.get("vehicles", [])):
                    st.session_state[f"carid_{i}"] = vehicle.get("id", f"Vehicle{i + 1}")
                    st.session_state[f"bcap_{i}"] = vehicle.get("battery_capacity", 0.0)
                    st.session_state[f"num_sessions_{i}"] = vehicle.get("num_sessions", 1)

                    for session_idx, session in enumerate(vehicle.get("sessions", [])):
                        # Parse dates and times
                        st.session_state[f"arr_date_{i}_{session_idx}"] = datetime.datetime.fromisoformat(
                            session.get("arrival_date", datetime.date.today().isoformat())).date()
                        st.session_state[f"arr_time_{i}_{session_idx}"] = datetime.datetime.fromisoformat(
                            session.get("arrival_time", datetime.time(0, 0).isoformat())).time()
                        st.session_state[f"dep_date_{i}_{session_idx}"] = datetime.datetime.fromisoformat(
                            session.get("departure_date", datetime.date.today().isoformat())).date()
                        st.session_state[f"dep_time_{i}_{session_idx}"] = datetime.datetime.fromisoformat(
                            session.get("departure_time", datetime.time(0, 0).isoformat())).time()
                        st.session_state[f"port_id_{i}_{session_idx}"] = session.get("port_id", 1)
                        st.session_state[f"port_label_{i}_{session_idx}"] = session.get("port_label",
                                                                                        "— Select a port —")

            # Load wind configuration
            if "wind_configuration" in config_data:
                wind_config = config_data["wind_configuration"]
                st.session_state["num_turbine_types"] = wind_config.get("num_turbine_types", 0)

                for i, turbine in enumerate(wind_config.get("turbines", [])):
                    st.session_state[f"hub_{i}"] = turbine.get("hub_height", 0.0)
                    st.session_state[f"count_{i}"] = turbine.get("number_of_turbines", 1)
                    st.session_state[f"turbine_id_{i}"] = turbine.get("turbine_id", "")

            # Load solar configuration
            if "solar_configuration" in config_data:
                solar_config = config_data["solar_configuration"]
                st.session_state["num_solar_types"] = solar_config.get("num_solar_types", 0)

                for i, panel in enumerate(solar_config.get("solar_panels", [])):
                    st.session_state[f"lat_{i}"] = panel.get("latitude", 0.0)
                    st.session_state[f"lon_{i}"] = panel.get("longitude", 0.0)
                    st.session_state[f"alt_{i}"] = panel.get("altitude", 0)
                    st.session_state[f"tilt_{i}"] = panel.get("surface_tilt", 0)
                    st.session_state[f"panels_{i}"] = panel.get("number_of_panels", 1)
                    st.session_state[f"module_id_{i}"] = panel.get("module_id", "")
                    st.session_state[f"inverter_id_{i}"] = panel.get("inverter_id", "")

            # Load plot settings
            if "plot_settings" in config_data:
                plot_settings = config_data["plot_settings"]
                st.session_state["plot_port"] = plot_settings.get("plot_port", 1)
                st.session_state["plot_car_id"] = plot_settings.get("plot_car_id", "")

        return True

    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return False


def get_saved_configurations():
    """
    Get list of all saved configuration files.
    """
    if not os.path.exists("saved_configurations"):
        return []

    config_files = []
    for filename in os.listdir("saved_configurations"):
        if filename.endswith(('.yaml', '.yml')):
            filepath = os.path.join("saved_configurations", filename)
            try:
                with open(filepath, 'r') as f:
                    config_data = yaml.safe_load(f)

                # Handle both new and legacy formats
                if "metadata" in config_data:
                    # New format
                    metadata = config_data["metadata"]
                    config_name = metadata.get("config_name", filename.replace('.yaml', '').replace('.yml', ''))
                    timestamp = metadata.get("timestamp", "")
                else:
                    # Legacy format
                    config_name = config_data.get("config_name", filename.replace('.yaml', '').replace('.yml', ''))
                    timestamp = config_data.get("timestamp", "")

                config_files.append({
                    "filename": filename,
                    "filepath": filepath,
                    "config_name": config_name,
                    "timestamp": timestamp
                })
            except:
                # If we can't read the file, just use the filename
                config_files.append({
                    "filename": filename,
                    "filepath": filepath,
                    "config_name": filename.replace('.yaml', '').replace('.yml', ''),
                    "timestamp": ""
                })

    # Sort by timestamp (newest first)
    config_files.sort(key=lambda x: x["timestamp"], reverse=True)
    return config_files


def delete_configuration_file(filepath):
    """
    Delete a configuration file.
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting configuration: {e}")
        return False


# ----------------------------- CONFIG LIST --------------------------

def create_config_component():
    st.header("💾 Saved Configurations")

    # Get list of saved configurations
    saved_configs = get_saved_configurations()

    if saved_configs:
        st.info(f"Found {len(saved_configs)} saved configuration(s):")

        for config in saved_configs:
            with st.expander(f"📁 {config['config_name']}", expanded=False):
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

                with col1:
                    st.write(f"**File:** {config['filename']}")
                    if config['timestamp']:
                        try:
                            dt = datetime.datetime.fromisoformat(config['timestamp'])
                            st.write(f"**Created:** {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                        except:
                            st.write(f"**Created:** {config['timestamp']}")

                with col2:
                    if st.button("📂 Load", key=f"load_{config['filename']}", type="secondary"):
                        if load_configuration_from_yaml(config['filepath']):
                            st.success(f"✅ Configuration '{config['config_name']}' loaded successfully!")
                            st.rerun()

                with col3:
                    if st.button("👁️ View", key=f"view_{config['filename']}", type="secondary"):
                        try:
                            with open(config['filepath'], 'r') as f:
                                content = f.read()
                            st.code(content, language="yaml")
                        except Exception as e:
                            st.error(f"Error reading file: {e}")

                with col4:
                    if st.button("🗑️ Delete", key=f"delete_{config['filename']}", type="primary"):
                        if delete_configuration_file(config['filepath']):
                            st.success(f"✅ Configuration '{config['config_name']}' deleted successfully!")
                            st.rerun()
    else:
        st.info("No saved configurations found. Save a configuration using the button below.")


# ----------------------------- READ CURRENT VALUES ------------------

def get_current_values():
    """Get current values from session state for simulation"""
    # Wind
    num_turbine_types = st.session_state.get("num_turbine_types", 1)
    turbines = []
    for i in range(int(num_turbine_types)):
        turbines.append({
            "hub_height": st.session_state.get(f"hub_{i}", 0.0),
            "number_of_turbines": st.session_state.get(f"count_{i}", 1),
            "turbine_id": st.session_state.get(f"turbine_id_{i}", ""),
        })

    # Solar
    num_solar_types = st.session_state.get("num_solar_types", 1)
    solar_panels = []
    for i in range(int(num_solar_types)):
        solar_panels.append({
            "latitude": st.session_state.get(f"lat_{i}", 0.0),
            "longitude": st.session_state.get(f"lon_{i}", 0.0),
            "altitude": st.session_state.get(f"alt_{i}", 0),
            "surface_tilt": st.session_state.get(f"tilt_{i}", 0),
            "number_of_panels": st.session_state.get(f"panels_{i}", 1),
            "inverter_id": st.session_state.get(f"inverter_id_{i}", ""),
            "module_id": st.session_state.get(f"module_id_{i}", ""),
        })

    # EVs - Updated to handle multi-session structure
    num_ev = st.session_state.get("num_ev", 0)
    ev_cars = []
    for i in range(int(num_ev)):
        num_sessions = st.session_state.get(f"num_sessions_{i}", 1)
        for session_idx in range(num_sessions):
            # Get battery capacity and ensure it's a float
            battery_cap = st.session_state.get(f"bcap_{i}", 0.0)
            if hasattr(battery_cap, 'item'):  # numpy scalar
                battery_cap = float(battery_cap.item())
            else:
                battery_cap = float(battery_cap)

            # Get dates and times
            arrival_date = st.session_state.get(f"arr_date_{i}_{session_idx}", datetime.date.today())
            arrival_time = st.session_state.get(f"arr_time_{i}_{session_idx}", datetime.time(0, 0))
            departure_date = st.session_state.get(f"dep_date_{i}_{session_idx}", datetime.date.today())
            departure_time = st.session_state.get(f"dep_time_{i}_{session_idx}", datetime.time(0, 0))

            ev_cars.append({
                "id": st.session_state.get(f"carid_{i}", f"Vehicle{i + 1}"),
                "battery_capacity": battery_cap,
                "charging_port": st.session_state.get(f"port_id_{i}_{session_idx}", 1),  # port ID
                "arrival_date": arrival_date.isoformat(),
                "arrival_time": arrival_time,
                "departure_date": departure_date.isoformat(),
                "departure_time": departure_time,
            })

    # System config
    storage_capacity = st.session_state.get("storage", 0.0)
    grid_capacity = st.session_state.get("grid", 0.0)
    initial_money = st.session_state.get("money", 0.000)
    price_high = st.session_state.get("price_high", 0.000)
    price_low = st.session_state.get("price_low", 0.000)
    timestep = st.session_state.get("timestep", 0)
    config_name = st.session_state.get("config_name", "")
    add_load = st.session_state.get("load", 0.0)

    # NEW: plot selections
    plot_port = int(st.session_state.get("plot_port", 1))
    plot_car_id = st.session_state.get("plot_car_id", "")  # string Vehicle ID / manual ID

    load_mode = st.session_state.get("load_mode", "constant")
    load_timeseries = st.session_state.get("load_timeseries", [])

    return (
        turbines,
        solar_panels,
        ev_cars,
        storage_capacity,
        grid_capacity,
        initial_money,
        price_high,
        price_low,
        config_name,
        add_load,  # legacy constant
        timestep,
        plot_port,
        plot_car_id,
        load_mode,  # <-- new
        load_timeseries  # <-- new
    )

