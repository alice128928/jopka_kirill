# components.py
import datetime
import os
import shutil

import streamlit as st
from streamlit import radio
import pandas as pd
from config_setup import delete_config, load_config

# -------------------------- YAML Upload Dialog --------------------------

# -------------------------- YAML Upload Dialog --------------------------

@st.dialog("📂 Load YAML Configs")
def yaml_loader_dialog():
    st.header("Upload Existing Configuration YAMLs")

    st.write("Upload the four configuration files. They will be saved as:")
    st.code("configurations/controller_add_config.yaml\n"
            "configurations/ev_config.yaml\n"
            "configurations/solar_config.yaml\n"
            "configurations/turbine_config.yaml", language="text")

    up_controller = st.file_uploader("controller_add_config.yaml", type=["yaml", "yml"], key="up_controller_yaml")
    up_ev         = st.file_uploader("ev_config.yaml",               type=["yaml", "yml"], key="up_ev_yaml")
    up_solar      = st.file_uploader("solar_config.yaml",            type=["yaml", "yml"], key="up_solar_yaml")
    up_turbine    = st.file_uploader("turbine_config.yaml",          type=["yaml", "yml"], key="up_turbine_yaml")

    def _validate_yaml(file_obj, required_top_keys=()):
        import yaml
        data = yaml.safe_load(file_obj.getvalue().decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("YAML root must be a mapping (dictionary).")
        for k in required_top_keys:
            if k not in data:
                raise ValueError(f"Missing top-level key: '{k}'")
        return data

    if st.button("Save Uploaded YAMLs", type="primary"):
        try:
            if not (up_controller and up_ev and up_solar and up_turbine):
                st.error("Please upload all four YAML files.")
                return

            os.makedirs("configurations", exist_ok=True)

            # Light checks
            _validate_yaml(up_controller, required_top_keys=("InitializationSettings",))
            _validate_yaml(up_ev,         required_top_keys=("InitializationSettings",))
            _validate_yaml(up_solar,      required_top_keys=("InitializationSettings",))
            _validate_yaml(up_turbine,    required_top_keys=("InitializationSettings",))

            # Save with expected names
            with open("configurations/controller_add_config.yaml", "wb") as f:
                f.write(up_controller.getbuffer())
            with open("configurations/ev_config.yaml", "wb") as f:
                f.write(up_ev.getbuffer())
            with open("configurations/solar_config.yaml", "wb") as f:
                f.write(up_solar.getbuffer())
            with open("configurations/turbine_config.yaml", "wb") as f:
                f.write(up_turbine.getbuffer())

            # Flag to run directly from uploaded YAMLs
            st.session_state["use_uploaded_yaml"] = True
            st.success("YAMLs saved. You can now run the simulation directly from the uploaded configs.")
            st.rerun()
        except Exception as e:
            st.error(f"Upload failed: {e}")


# ------------------------ Data Files Upload Dialog -----------------------

@st.dialog("📥 Upload Data Files")
def data_files_loader_dialog():
    """
    Upload EV and Load data files when not entered manually.
    Saves to 'data/' with expected filenames and stores paths in session_state.
    """
    st.header("Upload EV / Load Data Files")

    st.info("These uploads are needed **only** if you are not entering EVs or Load manually.")

    # Read current modes (from UI or uploaded YAML)
    option   = st.session_state.get("option", "")
    option_1 = st.session_state.get("option_1", "")

    os.makedirs("data", exist_ok=True)

    # ---------- EV files ----------
    st.subheader("EV Files")
    if option != "Enter data manually":
        st.write("EV mode is not manual → please upload:")
        up_cars = st.file_uploader("EV cars file (data_cars.csv/.xlsx)", type=["csv", "xlsx"], key="up_ev_cars")
        up_ports = st.file_uploader("Port capacity file (port_capacity_car.csv/.xlsx)", type=["csv", "xlsx"],
                                    key="up_ev_ports")

        if st.button("Save EV Files", key="btn_save_ev_files", type="primary"):
            if not up_cars or not up_ports:
                st.error("Please upload both EV files.")
            else:
                # Save with expected names & remember paths
                cars_ext = os.path.splitext(up_cars.name)[1].lower()
                ports_ext = os.path.splitext(up_ports.name)[1].lower()
                cars_path = os.path.join("data", f"data_cars{cars_ext}")
                ports_path = os.path.join("data", f"port_capacity_car{ports_ext}")
                with open(cars_path, "wb") as f:
                    f.write(up_cars.getbuffer())
                with open(ports_path, "wb") as f:
                    f.write(up_ports.getbuffer())
                st.session_state["cars_path"] = cars_path
                st.session_state["bcap_path"] = ports_path
                st.success(f"Saved: {cars_path} and {ports_path}")
                st.rerun()
    else:
        st.write("EV mode is manual → no EV file upload required.")

    st.markdown("---")

    # ---------- Load file ----------
    st.subheader("Load File")
    if option_1 != "Enter data manually":
        st.write("Load mode is not manual → please upload:")
        up_load = st.file_uploader("Load profile (LoadManyDays.csv/.xlsx)", type=["csv", "xlsx"], key="up_load_file")
        if st.button("Save Load File", key="btn_save_load_file", type="secondary"):
            if not up_load:
                st.error("Please upload the load profile file.")
            else:
                load_ext = os.path.splitext(up_load.name)[1].lower()
                load_path = os.path.join("data", f"LoadManyDays{load_ext}")
                with open(load_path, "wb") as f:
                    f.write(up_load.getbuffer())
                st.session_state["load_path"] = load_path
                st.success(f"Saved: {load_path}")
                st.rerun()
    else:
        st.write("Load mode is manual → no load file upload required.")

# ----------------------------- SETTINGS -----------------------------

@st.dialog("⚙️ Settings Configuration")
def settings_dialog():
    """Dialog for system configuration settings"""
    st.header("System Configuration")
    initial_money = st.number_input(
        "Initial budget ($)",
        min_value=0.0,
        format="%.4f",
        value=st.session_state.get("money", 0.0),
        key="dialog_money"
    )
    price_high = st.number_input(
        "Maximum price ($/kWh)",
        min_value=0.0,
        format="%.4f",
        value=st.session_state.get("price_high", 0.0),
        key="dialog_price_high"
    )
    price_low = st.number_input(
        "Minimum price ($/kWh)",
        min_value=0.0,
        format="%.4f",
        value=st.session_state.get("price_low", 0.0),
        key="dialog_price_low"
    )
    timestep = st.number_input(
        "Time step in the simulation (minutes)",
        min_value=0,
        format="%d",
        value=st.session_state.get("timestep", 0),
        key="dialog_timestep"
    )
    config_name = st.text_input(
        "Configuration version name",
        value=st.session_state.get("config_name", ""),
        key="dialog_config_name"
    )

    if st.button("Save Settings", type="primary"):
        st.session_state["money"] = initial_money
        st.session_state["price_high"] = price_high
        st.session_state["price_low"] = price_low
        st.session_state["timestep"] = timestep
        st.session_state["config_name"] = config_name
        st.success("Settings saved!")
        st.rerun()


# ----------------------------- BATTERY ------------------------------

@st.dialog("⚙️ Battery Configuration")
def battery_dialog():
    """Dialog for battery configuration"""
    st.header("Battery Configuration")
    storage_capacity = st.number_input(
        "Battery storage capacity (Wh)",
        min_value=0.0,
        value=st.session_state.get("storage", 0.0),
        key="dialog_storage"
    )

    if st.button("Save Settings", type="primary"):
        st.session_state["storage"] = storage_capacity
        st.success("Settings saved!")
        st.rerun()


# ----------------------------- GRID --------------------------------

@st.dialog("⚙️ Electrical Grid Configuration")
def grid_dialog():
    """Dialog for grid configuration"""
    st.header("Electrical Grid Configuration")
    grid_capacity = st.number_input(
        "Grid capacity (W)",
        min_value=0.0,
        value=st.session_state.get("grid", 0.0),
        key="dialog_grid"
    )

    if st.button("Save Settings", type="primary"):
        st.session_state["grid"] = grid_capacity
        st.success("Settings saved!")
        st.rerun()


# ----------------------------- EXTRA LOAD ---------------------------

@st.dialog("⚙️ Additional Load Configuration")
def load_dialog():
    """Dialog for additional load configuration"""
    st.header("Load Configuration")
    option_1 = st.selectbox(
        "How would you like to provide the load?",
        ("Enter data manually", "Upload a file"),
    )

    if option_1 == 'Enter data manually':
        add_load = st.number_input(
            "Load (W)",
            min_value=0.0,
            value=st.session_state.get("load", 0.0),
            key="dialog_load"
        )

        if st.button("Save Settings", type="primary"):
            st.session_state["load"] = add_load
            st.session_state["option_1"] = option_1
            st.success("Settings saved!")
            st.rerun()

    if option_1 == 'Upload a file':
        uploaded_files = st.file_uploader(
            "Choose a CSV/Excel file(s) with load profile", accept_multiple_files=True
        )
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            st.write("Original filename:", uploaded_file.name)

            if st.button("Save Load File", type="secondary", key=f"save_load_{uploaded_file.name}"):
                destination_folder = "data"
                os.makedirs(destination_folder, exist_ok=True)
                destination_path = os.path.join(destination_folder, "LoadManyDays.xlsx")
                st.session_state["option_1"] = option_1
                with open(destination_path, "wb") as f:
                    f.write(bytes_data)

                st.success(f"Load file saved as {destination_path}")
                st.rerun()


# ----------------------------- CHARGING PORTS -----------------------

@st.dialog("⚡ Charging Port Configuration")
def charging_dialog():
    """Dialog for charging ports configuration"""
    st.header("Charging Port Data")

    num_ports = st.number_input(
        "Number of charging ports",
        min_value=1,
        step=1,
        value=len(st.session_state.get("ports", [])) or 1,
        key="cp_num_ports",
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
            key=f"cp_port_cap_{i}",
        )

    st.markdown("---")
    # NEW: choose which port to plot (1-based)
    port_max = int(num_ports) if int(num_ports) > 0 else 1
    plot_port = st.number_input(
        "Port number to plot (1-based)",
        min_value=1,
        max_value=port_max,
        value=int(st.session_state.get("plot_port", 1)) if st.session_state.get("plot_port") else 1,
        step=1,
        key="cp_plot_port",
    )

    # save button
    if st.button("Save Ports", type="primary", key="cp_save_btn"):
        ports = []
        for i in range(int(num_ports)):
            ports.append({"capacity_W": float(st.session_state[f"cp_port_cap_{i}"])})
        st.session_state["ports"] = ports
        # persist selection
        st.session_state["plot_port"] = int(plot_port)
        st.success(f"Saved {len(ports)} charging port(s). Plotting Port {plot_port}.")
        st.rerun()


# ----------------------------- EVs ---------------------------------
@st.dialog("🔋 Electric Vehicle Configuration")
def ev_dialog():
    """Dialog for EV configuration"""
    st.header("Electric Vehicle Data")

    option = st.selectbox(
        "How would you like to provide EV data?",
        ("Enter data manually", "Upload a file"),
        key="ev_mode",
    )

    # ------------------ MANUAL MODE ------------------
    if option == "Enter data manually":
        num_ev = st.number_input(
            "Number of EVs",
            min_value=1,
            step=1,
            value=st.session_state.get("num_ev", 1),
            key="dialog_num_ev",
        )

        # Build selectable list of PHYSICAL ports (with ordinal)
        ports = st.session_state.get("ports", [])
        port_labels = ["— Select a port —"] + [
            f"Port {idx + 1} — {int(p['capacity_W'])} W" for idx, p in enumerate(ports)
        ]
        # map label -> (port_idx, capacity_W)
        label_to_info = {"— Select a port —": (-1, 0.0)}
        for idx, p in enumerate(ports):
            label_to_info[f"Port {idx + 1} — {int(p['capacity_W'])} W"] = (idx, p["capacity_W"])

        # Inputs for each EV
        for i in range(int(num_ev)):
            st.subheader(f"EV {i + 1}")
            st.text_input(
                f"Car ID for EV {i + 1}",
                value=st.session_state.get(f"carid_{i}", ""),
                key=f"dialog_carid_{i}",
            )
            st.number_input(
                f"Battery capacity (Wh) for EV {i + 1}",
                min_value=0.0,
                value=st.session_state.get(f"bcap_{i}", 0.0),
                key=f"dialog_bcap_{i}",
            )

            prev_label = st.session_state.get(f"port_label_{i}", "— Select a port —")
            st.selectbox(
                f"Assigned charging port for EV {i + 1}",
                port_labels,
                index=port_labels.index(prev_label) if prev_label in port_labels else 0,
                key=f"dialog_port_label_{i}",
            )

            st.time_input(
                f"Arrival time (HH:MM) for EV {i + 1}",
                value=st.session_state.get(f"arr_{i}", datetime.time(0, 0)),
                key=f"dialog_arr_{i}",
            )
            st.time_input(
                f"Departure time (HH:MM) for EV {i + 1}",
                value=st.session_state.get(f"dep_{i}", datetime.time(0, 0)),
                key=f"dialog_dep_{i}",
            )

        # Live “Car to plot” selector from currently visible IDs
        car_ids_live = [st.session_state.get(f"dialog_carid_{i}", f"EV {i+1}") for i in range(int(num_ev))]
        if car_ids_live:
            default_idx = 0
            if st.session_state.get("plot_car_id") in car_ids_live:
                default_idx = car_ids_live.index(st.session_state["plot_car_id"])
            chosen = st.selectbox("Car to plot", car_ids_live, index=default_idx, key="ev_plot_car_select")
            st.session_state["plot_car_id"] = chosen

        # Validate & Save
        if st.button("Save EV Configuration", type="primary", key="save_ev_manual"):
            # 1) Gather choices FIRST (do not save anything yet)
            selections = []  # (carid, port_idx, arr, dep, label, i, cap_W)
            for i in range(int(num_ev)):
                carid = st.session_state.get(f"dialog_carid_{i}", f"EV {i+1}")
                label = st.session_state.get(f"dialog_port_label_{i}", "— Select a port —")
                port_idx, cap_W = label_to_info.get(label, (-1, 0.0))
                arr = st.session_state.get(f"dialog_arr_{i}", datetime.time(0, 0))
                dep = st.session_state.get(f"dialog_dep_{i}", datetime.time(0, 0))
                selections.append((carid, port_idx, arr, dep, label, i, cap_W))

            # 2) Overlap test (same physical port only)
            def _to_min(t: datetime.time) -> int:
                return int(t.hour) * 60 + int(t.minute)

            def _expand(a: int, d: int):
                # convert a window into 1 or 2 segments in [0,1440)
                if a == d:  # treat as all-day
                    return [(0, 1440)]
                if a < d:
                    return [(a, d)]
                return [(a, 1440), (0, d)]  # wrap over midnight

            def _overlap(a1, d1, a2, d2) -> bool:
                s1 = _expand(_to_min(a1), _to_min(d1))
                s2 = _expand(_to_min(a2), _to_min(d2))
                for x1, y1 in s1:
                    for x2, y2 in s2:
                        if max(x1, x2) < min(y1, y2):  # strict intersection
                            return True
                return False

            conflicts = []
            for i in range(len(selections)):
                carA, portA, arrA, depA, labelA, idxA, _ = selections[i]
                if portA < 0:  # not selected
                    continue
                for j in range(i + 1, len(selections)):
                    carB, portB, arrB, depB, labelB, idxB, _ = selections[j]
                    if portA == portB and _overlap(arrA, depA, arrB, depB):
                        conflicts.append((labelA, carA, carB))

            if conflicts:
                lines = ["❌ The same port is assigned to multiple cars at overlapping times:"]
                for label, carA, carB in conflicts:
                    lines.append(f"• {label}: {carA} ↔ {carB}")
                st.error("\n".join(lines))
                return  # do NOT save

            # 3) Passed validation → now commit to session_state
            st.session_state["option"] = option
            st.session_state["num_ev"] = int(num_ev)
            for carid, port_idx, arr, dep, label, i, cap_W in selections:
                st.session_state[f"carid_{i}"] = st.session_state.get(f"dialog_carid_{i}", carid)
                st.session_state[f"bcap_{i}"] = st.session_state.get(f"dialog_bcap_{i}", 0.0)
                st.session_state[f"arr_{i}"] = arr
                st.session_state[f"dep_{i}"] = dep
                st.session_state[f"port_label_{i}"] = label
                st.session_state[f"port_{i}"] = float(cap_W)  # store ONLY capacity value (W)

            # Remember list of IDs and chosen car to plot
            st.session_state["all_car_ids"] = [s[0] for s in selections]
            if st.session_state.get("plot_car_id") not in st.session_state["all_car_ids"]:
                st.session_state["plot_car_id"] = st.session_state["all_car_ids"][0]

            st.success("EV configuration saved!")
            st.rerun()

    # ------------------ UPLOAD MODE ------------------
    else:
        car_file = st.file_uploader(
            "Upload EV cars file (e.g., data_cars.csv / .xlsx)",
            type=["csv", "xlsx"],
            key="ev_file_cars",
        )
        bcap_file = st.file_uploader(
            "Upload port capacity file (e.g., port_capacity_car.csv / .xlsx)",
            type=["csv", "xlsx"],
            key="ev_file_bcap",
        )
        function = st.radio(
            "Set which function you want to use", key="new", options=["predict soc", "predict end time"]
        )
        if car_file:
            st.write("EV cars file:", car_file.name)
        if bcap_file:
            st.write("Port capacity file:", bcap_file.name)
        if st.button("Save EV Configuration", type="secondary", key="save_ev_files_two"):
            if not car_file or not bcap_file:
                st.warning("Please upload both files before saving.")
            else:
                destination_folder = "data"
                os.makedirs(destination_folder, exist_ok=True)
                _, car_ext = os.path.splitext(car_file.name)
                _, bcap_ext = os.path.splitext(bcap_file.name)
                cars_path = os.path.join(destination_folder, f"data_cars{car_ext}")
                bcap_path = os.path.join(destination_folder, f"port_capacity_car{bcap_ext}")
                with open(cars_path, "wb") as f:
                    f.write(car_file.getbuffer())
                with open(bcap_path, "wb") as f:
                    f.write(bcap_file.getbuffer())
                st.session_state["option"] = "Upload a file"
                st.session_state["cars_path"] = cars_path
                st.session_state["bcap_path"] = bcap_path
                st.session_state["options"] = function
                st.success(f"Saved: {cars_path} and {bcap_path}")
                st.rerun()

        # Choose which Vehicle ID to plot (try to read from saved cars file)
        st.markdown("---")
        vehicle_ids = []
        cars_path = st.session_state.get("cars_path")
        if cars_path and os.path.exists(cars_path):
            try:
                if cars_path.lower().endswith(".csv"):
                    df = pd.read_csv(cars_path)
                else:
                    df = pd.read_excel(cars_path)
                # Look for plausible column names
                for col in ["Vehicle ID", "VehicleID", "Vehicle", "Veh ID"]:
                    if col in df.columns:
                        vehicle_ids = sorted(map(str, df[col].dropna().unique().tolist()))
                        break
            except Exception as e:
                st.warning(f"Could not read Vehicle IDs from file: {e}")

        if vehicle_ids:
            # Dropdown of IDs from file
            default_idx = 0
            if st.session_state.get("plot_car_id") in vehicle_ids:
                default_idx = vehicle_ids.index(st.session_state["plot_car_id"])
            chosen = st.selectbox(
                "Vehicle ID to plot (from uploaded file)",
                vehicle_ids,
                index=default_idx,
                key="plot_car_id_select",
            )
            st.session_state["plot_car_id"] = chosen
        else:
            # Fallback to free text
            st.text_input(
                "Vehicle ID to plot (type exact value from file, e.g., 'Vehicle#3')",
                value=st.session_state.get("plot_car_id", ""),
                key="plot_car_id",
            )

        # ===== NEW: Charger/Port IDs from the ports dataset (bcap/ports file) =====
        charger_ids = []
        bcap_path = st.session_state.get("bcap_path")
        if bcap_path and os.path.exists(bcap_path):
            try:
                if bcap_path.lower().endswith(".csv"):
                    dfp = pd.read_csv(bcap_path)
                else:
                    dfp = pd.read_excel(bcap_path)
                # Try common column names used for charger/port id
                for col in ["Ch ID", "ChID", "Ch_Id", "Ch", "ChId", "Port ID", "PortID", "Port"]:
                    if col in dfp.columns:
                        charger_ids = sorted(map(str, dfp[col].dropna().unique().tolist()))
                        break
            except Exception as e:
                st.warning(f"Could not read Charger IDs from ports file: {e}")

        if charger_ids:
            default_cidx = 0
            if st.session_state.get("plot_charger_id") in charger_ids:
                default_cidx = charger_ids.index(st.session_state["plot_charger_id"])
            chosen_c = st.selectbox(
                "Charger ID to plot (from uploaded ports file)",
                charger_ids,
                index=default_cidx,
                key="plot_charger_id_select",
            )
            st.session_state["plot_charger_id"] = chosen_c
        else:
            # Fallback to free text
            st.text_input(
                "Charger ID to plot (type exact 'Ch ID' from ports file)",
                value=st.session_state.get("plot_charger_id", ""),
                key="plot_charger_id",
            )


# ----------------------------- WIND --------------------------------

@st.dialog("💨 Wind Turbine Configuration")
def wind_dialog():
    """Dialog for wind turbine configuration"""
    st.header("Wind Turbine Data")
    num_turbine_types = st.number_input(
        "Number of wind farms",
        min_value=1,
        step=1,
        value=st.session_state.get("num_turbine_types", 1),
        key="dialog_num_turbine_types"
    )

    turbines = []
    for i in range(int(num_turbine_types)):
        st.subheader(f"Turbine farm {i+1}")
        hub_height = st.number_input(
            f"Hub height (m) for a turbine in farm {i+1}",
            min_value=0.0,
            value=st.session_state.get(f"hub_{i}", 0.0),
            key=f"dialog_hub_{i}"
        )

        number_of_turbines = st.number_input(
            f"Number of turbines in farm {i+1}",
            min_value=1,
            step=1,
            value=st.session_state.get(f"count_{i}", 1),
            key=f"dialog_count_{i}"
        )
        turbine_id = st.text_input(
            f"Turbine name (e.g., E-126/4200) for farm {i+1}",
            value=st.session_state.get(f"turbine_id_{i}", ""),
            key=f"dialog_turbine_id_{i}"
        )

        turbines.append({
            "hub_height": hub_height,
            "number_of_turbines": number_of_turbines,
            "turbine_id": turbine_id,
        })

    if st.button("Save Wind Configuration", type="primary"):
        st.session_state["num_turbine_types"] = num_turbine_types
        for i in range(int(num_turbine_types)):
            st.session_state[f"hub_{i}"] = st.session_state[f"dialog_hub_{i}"]
            st.session_state[f"count_{i}"] = st.session_state[f"dialog_count_{i}"]
            st.session_state[f"turbine_id_{i}"] = st.session_state[f"dialog_turbine_id_{i}"]
        st.success("Wind turbine configuration saved!")
        st.rerun()


# ----------------------------- SOLAR -------------------------------

@st.dialog("☀️ Solar Panel Configuration")
def solar_dialog():
    """Dialog for solar panel configuration"""
    st.header("Solar Panel Data")
    num_solar_types = st.number_input(
        "Number of solar farms",
        min_value=1,
        step=1,
        value=st.session_state.get("num_solar_types", 1),
        key="dialog_num_solar_types"
    )

    solar_panels = []
    for i in range(int(num_solar_types)):
        st.subheader(f"Solar farm {i+1}")
        latitude = st.number_input(
            f"Latitude (°) for panels in farm {i+1}",
            format="%.4f",
            min_value=-180.0,
            max_value=180.0,
            value=st.session_state.get(f"lat_{i}", 0.0),
            key=f"dialog_lat_{i}"
        )
        longitude = st.number_input(
            f"Longitude (°) for panels in farm {i+1}",
            format="%.4f",
            min_value=-180.0,
            max_value=180.0,
            value=st.session_state.get(f"lon_{i}", 0.0),
            key=f"dialog_lon_{i}"
        )
        altitude = st.number_input(
            f"Altitude (m) for panels in farm {i+1}",
            value=st.session_state.get(f"alt_{i}", 0),
            key=f"dialog_alt_{i}"
        )
        surface_tilt = st.number_input(
            f"Surface tilt (°) for panels in farm {i+1}",
            value=st.session_state.get(f"tilt_{i}", 0),
            key=f"dialog_tilt_{i}"
        )
        number_of_panels = st.number_input(
            f"Number of panels in farm {i+1}",
            min_value=1,
            step=1,
            value=st.session_state.get(f"panels_{i}", 1),
            key=f"dialog_panels_{i}"
        )
        module_id = st.text_input(
            f"Panel name for farm {i+1}",
            value=st.session_state.get(f"module_id_{i}", ""),
            key=f"dialog_module_id_{i}"
        )
        inverter_id = st.text_input(
            f"Inverter name for farm {i+1}",
            value=st.session_state.get(f"inverter_id_{i}", ""),
            key=f"dialog_inverter_id_{i}"
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

    if st.button("Save Solar Configuration", type="primary"):
        st.session_state["num_solar_types"] = num_solar_types
        for i in range(int(num_solar_types)):
            st.session_state[f"lat_{i}"] = st.session_state[f"dialog_lat_{i}"]
            st.session_state[f"lon_{i}"] = st.session_state[f"dialog_lon_{i}"]
            st.session_state[f"alt_{i}"] = st.session_state[f"dialog_alt_{i}"]
            st.session_state[f"tilt_{i}"] = st.session_state[f"dialog_tilt_{i}"]
            st.session_state[f"panels_{i}"] = st.session_state[f"dialog_panels_{i}"]
            st.session_state[f"inverter_id_{i}"] = st.session_state[f"dialog_inverter_id_{i}"]
            st.session_state[f"module_id_{i}"] = st.session_state[f"dialog_module_id_{i}"]

        st.success("Solar panel configuration saved!")
        st.rerun()


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

    # EVs
    num_ev = st.session_state.get("num_ev", 1)
    ev_cars = []
    for i in range(int(num_ev)):
        ev_cars.append({
            "id": st.session_state.get(f"carid_{i}", ""),
            "battery_capacity": st.session_state.get(f"bcap_{i}", 0.0),
            "charging_port": st.session_state.get(f"port_{i}", 0.0),  # capacity W
            "arrival_time": st.session_state.get(f"arr_{i}", datetime.time(0, 0)),
            "departure_time": st.session_state.get(f"dep_{i}", datetime.time(0, 0)),
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
    option = st.session_state.get("option", "")
    option_1 = st.session_state.get("option_1", "")
    options = st.session_state.get("options", "")


    # NEW: plot selections
    plot_port = int(st.session_state.get("plot_port", 1))
    plot_car_id = st.session_state.get("plot_car_id", "")  # string Vehicle ID / manual ID

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
        add_load,
        option,
        option_1,
        timestep,
        options,
        plot_port,
        plot_car_id,
    )

# ----------------------------- CONFIG LIST --------------------------

def create_config_component():
    st.header("Saved Configurations")
    if st.session_state.configs:
        st.info("You can load or delete saved configurations below:")
        cols = st.columns(3)
        for name in st.session_state.configs.keys():
            with cols[0]:
                st.write(f"Config name: **{name}**")
            with cols[1]:
                if st.button("Load", key=f"load_{name}", icon="💾", type="secondary"):
                    load_config(name)
            with cols[2]:
                if st.button("Delete", key=f"del_{name}", icon="🗑️", type="primary"):
                    delete_config(name)
    else:
        st.info("No saved configurations found. Create a new one below.")
