import streamlit as st


def init_configs():
    st.set_page_config(page_title="Co-simulation Input")
    if "configs" not in st.session_state:
        st.session_state.configs = {}
    if "current_config" not in st.session_state:
        st.session_state.current_config = None
    if "clicked_region" not in st.session_state:
        st.session_state.clicked_region = None


def load_config(name):
    cfg = st.session_state.configs[name]
    st.session_state.current_config = name
    # Wind turbines
    st.session_state.num_turbine_types = len(cfg["turbines"])
    for i, t in enumerate(cfg["turbines"]):
        st.session_state[f"hub_{i}"] = t["hub_height"]
        st.session_state[f"count_{i}"] = t["number_of_turbines"]
        st.session_state[f"turbine_id{i}"] = t["turbine_id"]
    # Solar panels
    st.session_state.num_solar_types = len(cfg["solar_panels"])
    for i, s in enumerate(cfg["solar_panels"]):
        st.session_state[f"lat_{i}"] = s["latitude"]
        st.session_state[f"lon_{i}"] = s["longitude"]
        st.session_state[f"alt_{i}"] = s["altitude"]
        st.session_state[f"tilt_{i}"] = s["surface_tilt"]
        st.session_state[f"panels_{i}"] = s["number_of_panels"]
        st.session_state[f"inverter_id_{i}"] = s["inverter_id"]
        st.session_state[f"module_id{i}"] = s["module_id"]
    # EVs
    st.session_state.num_ev = len(cfg["ev_cars"])
    for i, e in enumerate(cfg["ev_cars"]):
        st.session_state[f"carid_{i}"] = e["id"]
        st.session_state[f"bcap_{i}"] = e["battery_capacity"]
        st.session_state[f"port_{i}"] = e["charging_port"]
        st.session_state[f"arr_{i}"] = e["arrival_time"]
        st.session_state[f"dep_{i}"] = e["departure_time"]
    # System config
    sc = cfg["system_config"]
    st.session_state.storage = sc["storage_capacity"]
    st.session_state.grid = sc["grid_capacity"]
    st.session_state.money = sc["initial_money"]
    st.session_state.price_high = sc["price_high"]
    st.session_state.price_low = sc["price_low"]
    st.session_state.add_load = sc["add_load"]
    st.session_state.option = sc["option"]
    st.session_state.option_1 = sc["option_1"]
    st.session_state.timestep = sc["timestep"]
    st.session_state.options = sc["options"]

    st.session_state.config_name = name


def save_config(turbines, solar_panels, ev_cars,
                storage_capacity, grid_capacity, initial_money, price_high, price_low,add_load,option,option_1,timestep,options):
    # Gather all inputs
    cfg = {
        "turbines": turbines,
        "solar_panels": solar_panels,
        "ev_cars": ev_cars,
        "system_config": {
            "storage_capacity": storage_capacity,
            "grid_capacity": grid_capacity,
            "initial_money": initial_money,
            "price_high": price_high,
            "price_low": price_low,
            "add_load": add_load,
            "option": option,
            "option_1": option_1,
            "timestep": timestep,
            "options": options,
        }
    }
    name = st.session_state.config_name.strip()
    if not name:
        st.error("Please provide a name for this configuration before saving.")
        return
    st.session_state.configs[name] = cfg
    st.session_state.current_config = name
    st.success(f"Configuration '{name}' saved.")


def delete_config(name):
    if name in st.session_state.configs:
        del st.session_state.configs[name]
        if st.session_state.current_config == name:
            st.session_state.current_config = None
        st.success(f"Configuration '{name}' deleted.")
        st.rerun()
