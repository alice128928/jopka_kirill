# controller_ems_delta.py

import numpy as np
import yaml
from battery_storage import BatteryStorage  # adjust import path if needed

def controller_multiple_cars_2(
    status, power_charging_port, storage_obj, money,
    time_step, battery_capacity, solar, wind,
    current_price, storage_capacity, grid_capacity,
    voltage, power_set_point, ev_caps, availability, delta, constant_load_time,
    controller_settings,
):
    print(constant_load_time)
    """
    print(
    Main EMS logic for charging multiple EVs, handling energy flows, hub storage, and the grid.

    Units
    -----
    - Power: W
    - Energy: Wh
    - Price: $/Wh

    Sign convention for power_set_point (grid):
    -------------------------------------------
    +  import from grid (buying power)
    -  export to grid (selling power)

    Return
    ------
    money, storage_obj, power_set_point (float), battery_capacity
    """

    # -----------------------------
    # Timestep conversion & checks
    # -----------------------------
    delta_t_h = float(delta) / 60.0
    n_ev = len(status)
    # ───── Price thresholds from YAML ─────
    with open("configurations/controller_add_config.yaml", "r") as f:
        config_data = yaml.safe_load(f)
    configs = config_data["InitializationSettings"]["configs"][0]
    price_high = float(configs["price_high"])
    price_low = float(configs["price_low"])

    # -----------------------------
    # EVs requesting charge this step
    # -----------------------------
    charging_indices = [i for i in range(n_ev) if status[i][time_step] == 1 or status[i][time_step] == 5]

    # -----------------------------
    # Available renewable power this step
    # -----------------------------
    P_total_produced = float(solar + wind)  # W

    # ───── If no EVs are charging → price-based behavior ─────
    if not charging_indices:
        # Storage capability (W) this step
        SoC_Wh = float(storage_obj.get_soc())  # Wh
        room_Wh = float(storage_obj.get_remaining_capacity())  # Wh
        P_dis_cap = (SoC_Wh / delta_t_h) if delta_t_h > 0 else 0.0  # max discharge W
        P_chg_cap = (room_Wh / delta_t_h) if delta_t_h > 0 else 0.0  # max charge W

        # Defaults in case no storage action happens
        P_batt = 0.0
        P_battery_change = 0.0

        # Case 1: surplus renewables, low price → charge storage
        if (P_total_produced >= float(constant_load_time)) and (current_price < price_low):
            P_surplus = P_total_produced - float(constant_load_time)
            # Negative means "charge the battery"
            P_battery_change = -min(P_surplus, P_chg_cap)
            # storage.charge expects positive power; returns +W absorbed → negate to keep bus sign
            P_batt = -storage_obj.charge(-P_battery_change, delta_t_h)

        # Case 2: surplus renewables, high price → sell surplus (respect grid cap) and return
        if (P_total_produced >= float(constant_load_time)) and (current_price > price_high):
            P_surplus = P_total_produced - float(constant_load_time)
            # Clamp to grid capacity (export cannot exceed limit)
            export_W = min(P_surplus, grid_capacity)
            E_grid_Wh = abs(export_W) * delta_t_h
            money += E_grid_Wh * current_price
            return money, storage_obj, float(power_set_point), battery_capacity

        # Case 3: deficit, enough SoC → discharge storage
        if (P_total_produced <= float(constant_load_time)) and (SoC_Wh > 5000):
            P_deficit = float(constant_load_time) - P_total_produced
            P_battery_change = min(P_deficit, P_dis_cap)  # positive means "discharge the battery"
            P_batt = storage_obj.discharge(P_battery_change, delta_t_h)

        # Grid balance for the no-EV-charging case
        # If P_batt > 0 (discharge), it reduces grid import; if P_batt < 0 (charge), it increases export
        P_grid_unclamped = float(constant_load_time) - P_total_produced - P_batt
        P_grid = max(-grid_capacity, min(grid_capacity, P_grid_unclamped))

        # Money from grid trade
        E_grid_Wh = abs(P_grid) * delta_t_h
        if P_grid > 0:
            money -= E_grid_Wh * current_price  # import
        elif P_grid < 0:
            money += E_grid_Wh * current_price  # export

        return money, storage_obj, float(power_set_point), battery_capacity

    # -----------------------------
    # Hub storage capabilities for this step
    # (Assumes BatteryStorage API:
    #   - get_soc() -> available energy [Wh]
    #   - get_remaining_capacity() -> free room [Wh]
    #   - discharge(P_cmd_W, delta_t_h) -> actual delivered W over the step
    #   - charge(P_cmd_W, delta_t_h) -> actual absorbed W over the step)

    P_const = float(constant_load_time)            # W (background/house load)

    # EV requested power from port limits & energy room this step
    P_ev_req = 0.0
    P_req_by_ev = []
    for i in charging_indices:
        P_port = float(power_charging_port[i])     # W per EV
        E_room_Wh = max(0.0, float(ev_caps[i]) - float(battery_capacity[i][time_step]))
        P_room_W  = (E_room_Wh / delta_t_h) if delta_t_h > 0 else 0.0
        P_req     = min(P_port, P_room_W)
        P_req_by_ev.append((i, P_req))
        P_ev_req += P_req

    # Storage capability this step (convert Wh to W)
    SoC_Wh    = float(storage_obj.get_soc())
    room_Wh   = float(storage_obj.get_remaining_capacity())
    P_dis_cap = (SoC_Wh  / delta_t_h) if delta_t_h > 0 else 0.0
    P_chg_cap = (room_Wh / delta_t_h) if delta_t_h > 0 else 0.0

    # Decide storage action from the net *pre-grid* balance
    # net_pre_grid > 0  → deficit → discharge storage
    # net_pre_grid < 0  → surplus → charge storage
    net_pre_grid =  P_ev_req - P_total_produced
    if net_pre_grid > 0:
        P_battery_change_cmd = min(net_pre_grid, P_dis_cap)        # +W (discharge)
    elif net_pre_grid < 0:
        P_battery_change_cmd = -min(-net_pre_grid, P_chg_cap)      # -W (charge)
    else:
        P_battery_change_cmd = 0.0

    # Execute storage action (BatteryStorage uses power over delta_t_h)
    if P_battery_change_cmd >= 0:
        P_batt = storage_obj.discharge(P_battery_change_cmd, delta_t_h)   # +W to bus
    else:
        P_batt = -storage_obj.charge(-P_battery_change_cmd, delta_t_h)    # -W from bus

    # ---- Grid setpoint INCLUDING the constant load ----
    # Balance:  P_const + P_ev_used = P_total_produced + P_batt + P_grid
    # Use request first; clamping may curtail later
    P_grid_unclamped = P_const + P_ev_req - (P_total_produced + P_batt)
    P_grid = max(-grid_capacity, min(grid_capacity, P_grid_unclamped))    # + import, - export

    # ---- Power actually available for EVs after serving the constant load ----
    P_ev_avail = max(0.0, P_total_produced + P_batt + P_grid - P_const)

    # Proportional curtailment if limited
    ratio = 1.0 if P_ev_req <= 0 else max(0.0, min(1.0, P_ev_avail / P_ev_req))

    if time_step + 1 < len(battery_capacity[0]):
        for i, P_req in P_req_by_ev:
            P_alloc = P_req * ratio
            dE_Wh = P_alloc * delta_t_h
            battery_capacity[i][time_step + 1] = min(
                float(battery_capacity[i][time_step]) + dE_Wh,
                float(ev_caps[i])
            )

    # ---- Money from grid trade (only the net exchange) ----
    E_grid_Wh = abs(P_grid) * delta_t_h
    if P_grid > 0:
        money -= E_grid_Wh * current_price   # import
    elif P_grid < 0:
        money += E_grid_Wh * current_price   # export

    # Safety: ensure scalar return
    if isinstance(power_set_point, list):
        power_set_point = float(power_set_point[0])

    return money, storage_obj, float(10000), battery_capacity
