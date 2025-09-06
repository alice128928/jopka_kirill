import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
from datetime import datetime

from ev_request import ev_generate_from_config
from car_new import car_create
from car_new_soc import car_create_soc
from load_reading import give_load_w
from scipy.signal import find_peaks
from price_market import give_price
from battery_storage import BatteryStorage


# ---------- helpers ----------
def _resolve_car_idx(names, wanted_id: str) -> int:
    """Return index of wanted_id in names (exact match), else 0."""
    if not names:
        return 0
    wanted_id = (wanted_id or "").strip()
    for i, nm in enumerate(names):
        if str(nm).strip() == wanted_id:
            return i
    return 0


def port_power_series(charger_usage, charging_ports, selected_port, num_steps):
    """Return [W] time series for `selected_port` (1-based) from charger_usage."""
    series = []
    for t in range(num_steps):
        active = any(int(charger_usage[car][t]) == selected_port for car in range(len(charger_usage)))
        series.append(float(charging_ports[selected_port - 1]) if active else 0.0)
    return series


# ---------- light wrapper ----------
class Model:
    """Wrapper class for modeling any physical process (e.g., power flow, heat production)."""

    def __init__(self, process_model):
        if not callable(process_model):
            raise ValueError("The process must be a function or callable class.")
        self.process_model = process_model

    def calculate(self, *args):
        return self.process_model(*args)


class Manager:
    """Orchestrates the data exchanged between coupled models."""

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
        config = self.settings_configuration
        config_id = config["InitializationSettings"]["config_id"]
        start_time = config["InitializationSettings"]["time"]["start_time"]
        end_time = config["InitializationSettings"]["time"]["end_time"]

        grid_topology = pd.read_csv(config["InitializationSettings"]["grid_topology"])
        passive_consumer_power_setpoints = pd.read_csv(
            config["InitializationSettings"]["passive_consumers_power_setpoints"],
            index_col="snapshots",
            parse_dates=True,
        )

        # read controller YAML (plot selections, modes, etc.)
        with open("configurations/controller_add_config.yaml", "r") as f:
            config_data = yaml.safe_load(f)

        configs = config_data["InitializationSettings"]["configs"][0]
        grid_capacity = float(configs["grid_capacity"])
        initial_money = float(configs["initial_money"])
        storage_capacity = float(configs["storage_capacity"])
        option = configs["option"]                      # "Enter data manually" | "Upload a file"
        option_1 = configs["option_1"]                  # load mode
        delta_t = int(configs["timestep"])
        options = configs.get("options", "")            # predict mode when using files
        selected_port = int(configs.get("plot_port", 1))
        selected_car_id = str(configs.get("plot_car_id", "")).strip()

        # accumulators
        times = []
        wind_energy = []
        solar_energy = []
        power_setpoint_array = [0]
        storage_array = []
        storage_array_percentage = []
        money_array = [initial_money]
        signal_a = 0

        # This will hold the selected port power to plot (manual or predicted)
        selected_port_power = []

        battery = self.storage_battery.calculate(storage_capacity, 0.95, 0.95, 0)

        if option_1 == "Enter data manually":
            constant_load_value = float(configs["add_load"])
        else:
            constant_load_series = []
            signal_a = 1

        # EV inputs
        if option == "Enter data manually":
            car_names, battery_caps, charging_ports, availability_arrays, battery_capacity_arrays, status_cars = \
                ev_generate_from_config("configurations/ev_config.yaml", end_time, delta_t)
            charger_usage = None  # not used in manual mode
        else:
            if options == "predict end time":
                res = car_create_soc("data/data_cars.csv", "data/port_capacity_car.csv")
            else:  # "predict soc" default
                res = car_create("data/data_cars.csv", "data/port_capacity_car.csv")
            # expected 7 values
            (car_names, battery_caps, charger_usage, charging_ports,
             availability_arrays, battery_capacity_arrays, status_cars) = res

        # figure out which car to highlight
        car_idx_to_plot = _resolve_car_idx(car_names, selected_car_id)

        # main loop
        time_steps = int((end_time - start_time) / delta_t)
        for time_step in range(time_steps):
            time_clock = start_time + time_step * delta_t
            corresponding_time = passive_consumer_power_setpoints.index[time_step]

            # grid calc
            all_consumer_voltages = self.electric_grid.calculate(
                passive_consumer_power_setpoints,
                power_setpoint_array[time_step],
                grid_topology,
                corresponding_time,
            )

            # sources + market + EV state
            wind_energy_time = self.wind.calculate(time_clock)
            solar_energy_time = self.solar.calculate(time_clock)
            current_price_time = self.price_market.calculate(time_clock)
            ev_state_time = self.evstate.calculate(time_step, availability_arrays, battery_caps, battery_capacity_arrays)

            # load
            if signal_a == 1:
                constant_load_time = give_load_w(time_step, delta_t)
            else:
                constant_load_time = constant_load_value

            # update EV statuses
            for i in range(len(status_cars)):
                status_cars[i][time_step] = ev_state_time[i]

            # controller + selected-port power capture
            if option == "Enter data manually":
                out = self.controller_2.calculate(
                    status_cars, charging_ports, battery, money_array[time_step], time_step, battery_capacity_arrays,
                    solar_energy_time, wind_energy_time, current_price_time, storage_capacity, grid_capacity,
                    # smart_consumer_voltage not used for plotting anymore
                    all_consumer_voltages["consumers"]["smart_consumer"],
                    power_setpoint_array[time_step], battery_caps, availability_arrays,
                    delta_t, constant_load_time
                )
                # accept 4 or 5 results (5th often P_port of active charger)
                if isinstance(out, (list, tuple)):
                    if len(out) == 4:
                        money, battery, power_set_point, battery_capacity_arrays = out
                        P_port = 0.0
                    elif len(out) == 5:
                        money, battery, power_set_point, battery_capacity_arrays, P_port = out
                    else:
                        raise ValueError(f"controller_2.calculate returned {len(out)} values.")
                else:
                    raise ValueError("controller_2.calculate did not return a tuple/list.")

                # record manual selected-port power. If the controller gave P_port,
                # use it only when it corresponds to the chosen port; otherwise 0.
                # If your controller always returns the ACTIVE port's rating, we
                # can’t know which index it was; so we take P_port when it matches
                # the chosen port’s rating, else 0.
                sel_rating = float(charging_ports[selected_port - 1]) if charging_ports else 0.0
                selected_port_power.append(P_port if abs(P_port - sel_rating) < 1e-6 else 0.0)

            else:
                out = self.controller.calculate(
                    status_cars, charging_ports, battery, money_array[time_step], time_step, battery_capacity_arrays,
                    solar_energy_time, wind_energy_time, current_price_time, storage_capacity, grid_capacity,
                    all_consumer_voltages["consumers"]["smart_consumer"], power_setpoint_array[time_step],
                    battery_caps, availability_arrays, charger_usage, delta_t, constant_load_time, time_clock
                )
                # accept 4 or 5
                if isinstance(out, (list, tuple)):
                    if len(out) == 5:
                        money, battery, power_set_point, battery_capacity_arrays, P_port = out
                    elif len(out) == 4:
                        money, battery, power_set_point, battery_capacity_arrays = out
                        P_port = 0.0
                    else:
                        raise ValueError(f"controller.calculate returned {len(out)} values.")
                else:
                    raise ValueError("controller.calculate did not return a tuple/list.")

            # append state
            times.append(time_clock)
            wind_energy.append(wind_energy_time)
            solar_energy.append(solar_energy_time)
            if signal_a == 1:
                constant_load_series.append(constant_load_time)

            if isinstance(power_set_point, list):
                raise ValueError(f"Expected scalar power_set_point, got list: {power_set_point}")
            power_setpoint_array.append(power_set_point)
            storage_array.append(battery.get_soc())
            storage_array_percentage.append(battery.get_soc_percentage())
            money_array.append(money)

        # Build selected-port series for predicted mode from charger_usage;
        # for manual we already built it above. If manual didn’t produce anything,
        # fill with zeros so the plot renders.
        if option != "Enter data manually":
            selected_port_power = port_power_series(
                charger_usage=charger_usage,
                charging_ports=charging_ports,
                selected_port=selected_port,
                num_steps=len(times),
            )
        if not selected_port_power:
            selected_port_power = [0.0] * len(times)

        # choose plotter
        if option == "Enter data manually":
            self.plot_results_manual(
                times,
                selected_port_power,             # <— plot selected port (not voltage)
                battery_capacity_arrays,
                solar_energy,
                wind_energy,
                storage_array,
                money_array,
                config_id,
                car_idx_to_plot,                 # <— only selected car
            )
        else:
            self.plot_results_pred(
                times,
                selected_port_power,
                battery_capacity_arrays,
                solar_energy,
                wind_energy,
                storage_array,
                money_array,
                config_id,
                charger_usage,
                charging_ports,
                car_idx_to_plot,                 # <— only selected car
            )

    # --------------------- plotters ---------------------
    def plot_results_manual(self, times, port_power, battery_capacity_arrays, solar_energy, wind_energies,
                            storage_array, money_array, config_id, car_idx_to_plot: int):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.signal import find_peaks

        series_1d = [port_power, solar_energy, wind_energies, storage_array, money_array]
        min_len = min([len(times)] + [len(s) for s in series_1d])
        times = np.asarray(times)[:min_len]
        port_power = np.asarray(port_power)[:min_len]
        solar_energy = np.asarray(solar_energy)[:min_len]
        wind_energies = np.asarray(wind_energies)[:min_len]
        storage_array = np.asarray(storage_array)[:min_len]
        money_array = np.asarray(money_array)[:min_len]
        battery_capacity_arrays = [np.asarray(b)[:min_len] for b in battery_capacity_arrays]

        fig, axs = plt.subplots(3, 2, figsize=(14, 10))

        # Selected Port Power (top-left)
        axs[0, 0].plot(times, port_power)
        axs[0, 0].set_title("Selected Port Power", color="black")
        axs[0, 0].set_xlabel("Time [steps]", color="black")
        axs[0, 0].set_ylabel("Power [W]", color="black")

        # Selected Car’s Battery Capacity (top-right)
        car_capacity = battery_capacity_arrays[car_idx_to_plot]
        peaks, _ = find_peaks(car_capacity, height=0)
        axs[0, 1].plot(times, car_capacity, label=f"Car {car_idx_to_plot + 1}")
        if len(peaks) > 0:
            axs[0, 1].plot(times[peaks], car_capacity[peaks], "o", ms=3)
        axs[0, 1].set_title("Battery Capacity (Selected Car)", color="black")
        axs[0, 1].set_xlabel("Time [steps]", color="black")
        axs[0, 1].set_ylabel("Capacity [Wh]", color="black")
        axs[0, 1].legend(fontsize=8)

        # Solar
        axs[1, 0].plot(times, solar_energy)
        axs[1, 0].set_title("Solar Production", color="black")
        axs[1, 0].set_xlabel("Time [steps]", color="black")
        axs[1, 0].set_ylabel("Energy [Wh]", color="black")

        # Wind
        axs[1, 1].plot(times, wind_energies)
        axs[1, 1].set_title("Wind Energy", color="black")
        axs[1, 1].set_xlabel("Time [steps]", color="black")
        axs[1, 1].set_ylabel("Energy [Wh]", color="black")

        # Storage
        axs[2, 0].plot(times, storage_array)
        axs[2, 0].set_title("Storage", color="black")
        axs[2, 0].set_xlabel("Time [steps]", color="black")
        axs[2, 0].set_ylabel("Energy [Wh]", color="black")

        # Money
        axs[2, 1].plot(times, money_array)
        axs[2, 1].set_title("Money", color="black")
        axs[2, 1].set_xlabel("Time [steps]", color="black")
        axs[2, 1].set_ylabel("Money [$]", color="black")

        plt.tight_layout()
        plt.savefig(f"results_config{config_id}.png")

    def plot_results_pred(self, times, port_power, battery_capacity_arrays, solar_energy, wind_energies,
                          storage_array, money_array, config_id, charger_usage, power_charging_port,
                          car_idx_to_plot: int):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.signal import find_peaks

        series_1d = [port_power, solar_energy, wind_energies, storage_array, money_array]
        min_len = min([len(times)] + [len(s) for s in series_1d])
        times = np.asarray(times)[:min_len]
        port_power = np.asarray(port_power)[:min_len]
        solar_energy = np.asarray(solar_energy)[:min_len]
        wind_energies = np.asarray(wind_energies)[:min_len]
        storage_array = np.asarray(storage_array)[:min_len]
        money_array = np.asarray(money_array)[:min_len]
        battery_capacity_arrays = [np.asarray(b)[:min_len] for b in battery_capacity_arrays]

        plt.style.use("ggplot")
        fig, axs = plt.subplots(3, 2, figsize=(14, 10))

        # Selected Port Power
        axs[0, 0].plot(times, port_power)
        axs[0, 0].set_title("Selected Port Power", color="black")
        axs[0, 0].set_xlabel("Time [hours]", color="black")
        axs[0, 0].set_ylabel("Power [W]", color="black")

        # Selected Car’s Battery Capacity (with charger labels)
        car_capacity = battery_capacity_arrays[car_idx_to_plot]
        peaks, _ = find_peaks(car_capacity, height=0)
        peak_values = car_capacity[peaks]
        axs[0, 1].plot(times, car_capacity, label=f"Car {car_idx_to_plot + 1}", color="black")
        axs[0, 1].plot(times[peaks], peak_values, "ro", label="Peaks")
        for tp, val in zip(peaks, peak_values):
            ch = int(charger_usage[car_idx_to_plot][tp]) if charger_usage is not None else 0
            label = f"#{ch}"
            axs[0, 1].annotate(label, (times[tp], val), xytext=(4, 8),
                               textcoords="offset points", fontsize=8)
        axs[0, 1].set_title("Battery Capacity (Selected Car)", color="black")
        axs[0, 1].set_xlabel("Time [hours]", color="black")
        axs[0, 1].set_ylabel("Battery Capacity [Wh]", color="black")
        axs[0, 1].legend()

        # Solar
        axs[1, 0].plot(times, solar_energy, color="orange")
        axs[1, 0].set_title("Solar Production", color="black")
        axs[1, 0].set_xlabel("Time [hours]", color="black")
        axs[1, 0].set_ylabel("Solar Energy [Wh]", color="black")

        # Wind
        axs[1, 1].plot(times, wind_energies, color="purple")
        axs[1, 1].set_title("Wind Energy", color="black")
        axs[1, 1].set_xlabel("Time [hours]", color="black")
        axs[1, 1].set_ylabel("Wind Energy [Wh]", color="black")

        # Storage
        axs[2, 0].plot(times, storage_array, color="green")
        axs[2, 0].set_title("Storage", color="black")
        axs[2, 0].set_xlabel("Time [hours]", color="black")
        axs[2, 0].set_ylabel("Storage [Wh]", color="black")

        # Money
        axs[2, 1].plot(times, money_array, color="red")
        axs[2, 1].set_title("Money", color="black")
        axs[2, 1].set_xlabel("Time [hours]", color="black")
        axs[2, 1].set_ylabel("Money [$]", color="black")

        plt.tight_layout()
        plt.savefig(f"results_config{config_id}.png")
