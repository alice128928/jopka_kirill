# app.py

import os
from datetime import time
import yaml
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates



from components import (
    create_config_component,
    ev_dialog,
    get_current_values,
    settings_dialog,
    solar_dialog,
    wind_dialog,
    grid_dialog,
    battery_dialog,
    load_dialog,
    charging_dialog,
    yaml_loader_dialog,
    data_files_loader_dialog,
)
from config_setup import init_configs, save_config
from run_simulation import run


# ----------------------------- Helpers -----------------------------

def get_rectangle_coords(corners):
    """Convert two points to rectangle coords (x1, y1, x2, y2) with correct ordering."""
    (x1, y1), (x2, y2) = corners
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

def point_in_rectangle(point, rectangle_corners):
    """Check if point (x, y) lies within a rectangle defined by two corner points."""
    x, y = point
    (x1, y1), (x2, y2) = rectangle_corners
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    return (min_x <= x <= max_x) and (min_y <= y <= max_y)

def check_click_in_regions(click_point, regions):
    """Return region name if click_point is inside any region, else None."""
    for region_name, coords in regions.items():
        if point_in_rectangle(click_point, coords):
            return region_name
    return None


# Small wrappers so we can set flags when dialogs are opened
def open_yaml_loader_dialog():
    st.session_state["used_yaml_loader"] = True
    yaml_loader_dialog()

def open_data_files_loader_dialog():
    data_files_loader_dialog()


# --------------------------- App bootstrap --------------------------

st.set_page_config(page_title="Co-simulation Input Parameters", layout="wide")
init_configs()

# Session state defaults
if "result_ready" not in st.session_state:
    st.session_state["result_ready"] = False
if "last_click" not in st.session_state:
    st.session_state["last_click"] = None
if "used_yaml_loader" not in st.session_state:
    st.session_state["used_yaml_loader"] = False
if "save_config_clicked" not in st.session_state:
    st.session_state["save_config_clicked"] = False

st.title("Co-simulation Input Parameters")
create_config_component()
st.divider()
st.button("📂 Load YAML Configs", on_click=open_yaml_loader_dialog, type="secondary")
st.button("📥 Upload Data Files", on_click=open_data_files_loader_dialog, type="secondary")

# --------------------------- Image + Regions ------------------------

# Load and prepare the base image every run (Streamlit re-runs on interactions)
size = (700, 700)
img = Image.open("power_new_2.png").resize(size, Image.Resampling.LANCZOS)
draw = ImageDraw.Draw(img)

# Define rectangle corners for each component (top-left & bottom-right-ish points)
settings_coordinates = [(250, 320), (460, 140)]
ev_coordinates       = [(470, 400), (610, 330)]
wind_coordinates     = [(490, 310), (670,  50)]
solar_coordinates    = [( 70, 230), (220,  80)]
house_coordinates    = [( 50, 630), (230, 430)]
grid_coordinates     = [(450, 530), (290, 690)]
battery_coordinates  = [(120, 380), (220, 260)]
port_coordinates     = [(600, 530), (440, 400)]

regions = {
    "Settings": settings_coordinates,
    "EV": ev_coordinates,
    "Wind": wind_coordinates,
    "Solar": solar_coordinates,
    "House": house_coordinates,
    "Grid": grid_coordinates,
    "Battery": battery_coordinates,
    "Port": port_coordinates,
}

# Draw the rectangles
draw.rectangle(get_rectangle_coords(settings_coordinates), outline="blue",   width=2)
draw.rectangle(get_rectangle_coords(ev_coordinates),       outline="green",  width=2)
draw.rectangle(get_rectangle_coords(wind_coordinates),     outline="orange", width=2)
draw.rectangle(get_rectangle_coords(solar_coordinates),    outline="yellow", width=2)
draw.rectangle(get_rectangle_coords(house_coordinates),    outline="pink",   width=2)
draw.rectangle(get_rectangle_coords(grid_coordinates),     outline="black",  width=2)
draw.rectangle(get_rectangle_coords(battery_coordinates),  outline="red",    width=2)
draw.rectangle(get_rectangle_coords(port_coordinates),     outline="green",  width=2)

st.info("💡 **Instructions:** Click inside a colored rectangle to open its dialog. You can keep clicking even after running a simulation.")

# ------------------------- Layout: two columns ----------------------

left, right = st.columns([1, 1], gap="large")

# Always render the interactive image (left column)
with left:
    clicked_coordinates = streamlit_image_coordinates(
        img,
        key="power_topology_clickmap",
        width=size[0],
        height=size[1],
    )

    if clicked_coordinates and ("x" in clicked_coordinates) and ("y" in clicked_coordinates):
        click_point = (clicked_coordinates["x"], clicked_coordinates["y"])

        # Avoid triggering the same click twice if the component re-reports the last coordinates
        if st.session_state["last_click"] != click_point:
            st.session_state["last_click"] = click_point

            clicked_region = check_click_in_regions(click_point, regions)
            if clicked_region:
                if clicked_region == "Settings":
                    settings_dialog()
                elif clicked_region == "EV":
                    ev_dialog()
                elif clicked_region == "Wind":
                    wind_dialog()
                elif clicked_region == "Solar":
                    solar_dialog()
                elif clicked_region == "Grid":
                    grid_dialog()
                elif clicked_region == "Battery":
                    battery_dialog()
                elif clicked_region == "House":
                    load_dialog()
                elif clicked_region == "Port":
                    charging_dialog()

st.divider()

# ------------------- Controls: Save + Run buttons -------------------

(turbines, solar_panels, ev_cars, storage_capacity, grid_capacity,
 initial_money, price_high, price_low, config_name,
 add_load, option, option_1, timestep, options,
 plot_port, plot_car_id) = get_current_values()

controls_col1, controls_col2 = st.columns([1, 1])

with controls_col1:
    if st.button("💾 Save Configuration"):
        if config_name:
            # Ensure times are YAML-safe strings
            for car in ev_cars:
                if isinstance(car.get("arrival_time"), time):
                    car["arrival_time"] = car["arrival_time"].strftime("%H:%M")
                if isinstance(car.get("departure_time"), time):
                    car["departure_time"] = car["departure_time"].strftime("%H:%M")

            # Build YAML config blobs
            system_config = [{
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
                # NEW:
                "plot_port": int(plot_port),
                "plot_car_id": str(plot_car_id or ""),
            }]

            wind_config_data = {
                "InitializationSettings": {"config_id": "config 1"},
                "wind_turbines": turbines,
            }
            solar_config_data = {
                "InitializationSettings": {"config_id": "config 1"},
                "solar_panels": solar_panels,
            }
            ev_config_data = {
                "InitializationSettings": {"config_id": "config 1", "cars": ev_cars}
            }
            controller_config_add = {
                "InitializationSettings": {"config_id": "config 1", "configs": system_config}
            }

            os.makedirs("configurations", exist_ok=True)
            with open("configurations/turbine_config.yaml", "w") as f:
                yaml.dump(wind_config_data, f, default_flow_style=False)
            with open("configurations/solar_config.yaml", "w") as f:
                yaml.dump(solar_config_data, f, default_flow_style=False)
            with open("configurations/ev_config.yaml", "w") as f:
                yaml.dump(ev_config_data, f, default_flow_style=False)
            with open("configurations/controller_add_config.yaml", "w") as f:
                yaml.dump(controller_config_add, f, default_flow_style=False)

            # Optional: persist through your own helper as well
            save_config(turbines, solar_panels, ev_cars,
                        storage_capacity, grid_capacity,
                        initial_money, price_high, price_low,
                        add_load, option, option_1, timestep, options)

            st.session_state["save_config_clicked"] = True
            st.success("Configuration saved ✅")
        else:
            st.error("Please set a configuration name in the Settings dialog first!")
            settings_dialog()

with controls_col2:
    if st.button("▶️ Run Simulation", type="primary"):
        st.info("Running simulation with current inputs...")
        try:
            use_uploaded_yaml = st.session_state.get("use_uploaded_yaml", False)
            save_config_clicked = st.session_state.get("save_config_clicked", False)
            used_yaml_loader = st.session_state.get("used_yaml_loader", False)

            # ---------- Validate required data files when not manual ----------
            def _exists(path):
                return path and os.path.exists(path)

            # EV data
            if option != "Enter data manually":
                cars_path = st.session_state.get("cars_path")
                bcap_path = st.session_state.get("bcap_path")
                # If paths not in session_state, try defaults
                if not _exists(cars_path):
                    for ext in (".csv", ".xlsx"):
                        p = os.path.join("data", f"data_cars{ext}")
                        if os.path.exists(p):
                            cars_path = p; st.session_state["cars_path"] = p; break
                if not _exists(bcap_path):
                    for ext in (".csv", ".xlsx"):
                        p = os.path.join("data", f"port_capacity_car{ext}")
                        if os.path.exists(p):
                            bcap_path = p; st.session_state["bcap_path"] = p; break
                if not (_exists(cars_path) and _exists(bcap_path)):
                    st.error("EV mode is not manual, but EV files are missing. "
                             "Click **📥 Upload Data Files** and upload both EV files.")
                    st.stop()

            # Load data
            if option_1 != "Enter data manually":
                load_path = st.session_state.get("load_path")
                if not _exists(load_path):
                    for ext in (".csv", ".xlsx"):
                        p = os.path.join("data", f"LoadManyDays{ext}")
                        if os.path.exists(p):
                            load_path = p; st.session_state["load_path"] = p; break
                if not _exists(load_path):
                    st.error("Load mode is not manual, but Load file is missing. "
                             "Click **📥 Upload Data Files** and upload the load profile.")
                    st.stop()

            # ---------- Write YAMLs only if we're in UI-config mode AND user didn't click Save ----------
            # Interpretation of your request:
            # If the user did NOT click "Load YAML Configs" and did NOT click "Save Configuration",
            # then we still write YAMLs from the current UI values before running.
            if (not use_uploaded_yaml) and (not save_config_clicked):
                # Ensure times are YAML-safe strings
                for car in ev_cars:
                    if isinstance(car.get("arrival_time"), time):
                        car["arrival_time"] = car["arrival_time"].strftime("%H:%M")
                    if isinstance(car.get("departure_time"), time):
                        car["departure_time"] = car["departure_time"].strftime("%H:%M")

                system_config = [{
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
                    # NEW:
                    "plot_port": int(plot_port),
                    "plot_car_id": str(plot_car_id or ""),
                }]

                wind_config_data = {
                    "InitializationSettings": {"config_id": "config 1"},
                    "wind_turbines": turbines,
                }
                solar_config_data = {
                    "InitializationSettings": {"config_id": "config 1"},
                    "solar_panels": solar_panels,
                }
                ev_config_data = {
                    "InitializationSettings": {"config_id": "config 1", "cars": ev_cars}
                }
                controller_config_add = {
                    "InitializationSettings": {"config_id": "config 1", "configs": system_config}
                }

                os.makedirs("configurations", exist_ok=True)
                with open("configurations/turbine_config.yaml", "w") as f:
                    yaml.dump(wind_config_data, f, default_flow_style=False)
                with open("configurations/solar_config.yaml", "w") as f:
                    yaml.dump(solar_config_data, f, default_flow_style=False)
                with open("configurations/ev_config.yaml", "w") as f:
                    yaml.dump(ev_config_data, f, default_flow_style=False)
                with open("configurations/controller_add_config.yaml", "w") as f:
                    yaml.dump(controller_config_add, f, default_flow_style=False)

                st.info("Wrote YAMLs from current UI values (auto, no Save clicked).")
            else:
                if use_uploaded_yaml or used_yaml_loader:
                    st.info("Using uploaded YAMLs from the configurations/ folder.")
                elif save_config_clicked:
                    st.info("Using YAMLs saved via '💾 Save Configuration'.")

            # ---------- Run ----------
            with st.spinner("Simulating..."):
                run()

            st.session_state["result_ready"] = True
            st.success("Simulation finished ✅")
        except Exception as e:
            st.session_state["result_ready"] = False
            st.error(f"Simulation failed: {e}")


st.divider()

# ------------------------- Results (always) -------------------------

with right:
    if st.session_state.get("result_ready", False):
        result_path = "results_config1.png"
        if os.path.exists(result_path):
            st.image(result_path, use_container_width=True, caption="Final result")
        else:
            st.warning(f"Result image '{result_path}' not found.")
    else:
        st.info("Run a simulation to see results here.")

# ----------------------------- End ---------------------------------
