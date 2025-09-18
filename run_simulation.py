# run_simulation.py
from functools import partial

from controller_ems import controller_multiple_cars
from controller_2_ems import controller_multiple_cars_2
from cosim_framework import Manager, Model
from grid import electric_grid_function
from load_configurations import load_configurations
from price_market import give_price
from battery_storage import BatteryStorage

def run():
    configurations_folder_path = "configurations"
    controller_config, settings_configs = load_configurations(configurations_folder_path)
    shared_config = settings_configs["config 1"]

    from wind_plant import power_wind
    from solar_panel import solar_Power
    from ev_battery_state import ev_state

    electric_grid_model = Model(electric_grid_function)
    controller_model = Model(partial(controller_multiple_cars, controller_settings=controller_config))
    controller_model_2 = Model(partial(controller_multiple_cars_2, controller_settings=controller_config))
    wind_plant_model = Model(power_wind)
    solar_panel_model = Model(solar_Power)
    ev_model = Model(ev_state)
    price_market_model = Model(give_price)
    storage_battery_model = Model(BatteryStorage)

    models = [
        electric_grid_model,
        controller_model,
        wind_plant_model,
        solar_panel_model,
        ev_model,
        price_market_model,
        storage_battery_model,
        controller_model_2,
    ]

    manager = Manager(models, shared_config)

    # Return dict: {"timeseries": pd.DataFrame, "soc": pd.DataFrame}
    return manager.run_simulation()
