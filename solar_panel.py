import yaml
import numpy as np
import pandas as pd
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem, retrieve_sam
from pvlib.location import Location
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

# --- Global cache ---
_weather_data = None
_pv_systems = None
_locations = None
_number_of_panels_list = None


def get_weather_data(filename='data/data_sun.xlsx'):
    df = pd.read_excel(filename).drop(0)
    df.columns = df.iloc[0]
    df = df.drop(1)

    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df.set_index('datetime', inplace=True)
    df.index = pd.to_datetime(df.index)

    df = df.rename(columns={
        "Temperature": "temp_air",
        "Wind Speed": "wind_speed",
        "Relative Humidity": "humidity",
        "Precipitable Water": "precipitable_water",
        "GHI": "ghi",
        "DNI": "dni",
        "DHI": "dhi"
    })

    for col in ['temp_air', 'wind_speed', 'humidity', 'precipitable_water', 'ghi', 'dni', 'dhi']:
        df[col] = df[col].astype(float)

    df = df[['temp_air', 'wind_speed', 'humidity', 'precipitable_water', 'ghi', 'dni', 'dhi']]
    return df.resample('1h').mean().iloc[:480]


def load_panel_config(path='configurations/solar_config.yaml'):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get("solar_panels", [])


def create_pv_system(panel_data: dict):
    module_db = retrieve_sam("cecmod")
    inverter_db = retrieve_sam("cecinverter")

    module_id = panel_data["module_id"].strip()
    inverter_id = panel_data["inverter_id"].strip()

    # --- Validate entries
    if module_id not in module_db.columns:
        raise ValueError(f"Module ID '{module_id}' not found in CEC module database.")

    if inverter_id not in inverter_db.columns:
        raise ValueError(f"Inverter ID '{inverter_id}' not found in CEC inverter database.")

    module = module_db[module_id]
    inverter = inverter_db[inverter_id]

    temp_model = TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]

    location = Location(
        latitude=panel_data["latitude"],
        longitude=panel_data["longitude"],
        altitude=panel_data["altitude"]
    )

    system = PVSystem(
        surface_tilt=panel_data["surface_tilt"],
        surface_azimuth=180,
        module_parameters=module,
        inverter_parameters=inverter,
        temperature_model_parameters=temp_model
    )

    return system, location, panel_data["number_of_panels"]


def init_solar_model():
    global _weather_data, _pv_systems, _locations, _number_of_panels_list

    if _weather_data is None:
        _weather_data = get_weather_data()

    panel_config = load_panel_config()

    _pv_systems = []
    _locations = []
    _number_of_panels_list = []

    for panel_data in panel_config:
        system, location, number_of_panels = create_pv_system(panel_data)
        _pv_systems.append(system)
        _locations.append(location)
        _number_of_panels_list.append(number_of_panels)


def solar_Power(time_index: int) -> float:
    """
    Return solar power at a given minute index.
    Internally, weather/PV data is hourly, so each hour's value
    is repeated for 60 minutes.
    """
    global _weather_data, _pv_systems, _locations, _number_of_panels_list

    if time_index < 0:
        return 0.0

    # map minute index to hourly index
    hour_index = time_index // 60

    if hour_index >= len(_weather_data):
        return 0.0

    total_power = 0.0
    weather_step = _weather_data.iloc[[hour_index]]

    for system, location, num_panels in zip(_pv_systems, _locations, _number_of_panels_list):
        mc = ModelChain(system, location, aoi_model="physical")
        mc.run_model(weather=weather_step)
        ac_output = mc.results.ac.iloc[0] if not mc.results.ac.empty else 0.0
        total_power += ac_output * num_panels

    return total_power

# Initialize once on import
init_solar_model()
