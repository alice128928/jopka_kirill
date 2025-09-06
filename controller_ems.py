import yaml
from battery_storage import BatteryStorage  # keep import if your project expects it

import pandas as pd

# ---- Load once ----
DF = pd.read_excel("data/port_capacity_car.xlsx")
DF = DF.rename(columns={
    "Ch ID": "ch_id",
    "Time [ISO8601]": "time_iso",
    "Predicted output power [kW]": "p_pred_kw"
})
DF["time_iso"] = pd.to_datetime(DF["time_iso"]).dt.floor("min")

# Per-port minute index
PORT_INDEX = {
    port: d.sort_values("time_iso").set_index("time_iso")[["p_pred_kw"]]
    for port, d in DF.groupby("ch_id")
}

# Simulation start = midnight of earliest day in file
SIM_START_TS = DF["time_iso"].min().normalize()


def get_predicted_power_W_by_minute(port_number: int,
                                    minute_offset: int,
                                    *,
                                    allow_nearest: bool = True,
                                    clamp_edges: bool = False,
                                    column_is_kW: bool = False) -> float:
    """
    Lookup predicted charger power at a given minute offset.
    Returns power in **Watts**.

    port_number   : Ch ID from the sheet
    minute_offset : minutes since SIM_START_TS (00:00 of earliest date)
    allow_nearest : use nearest minute if exact match missing
    clamp_edges   : when outside data range, use first/last value (else 0)
    column_is_kW  : set True if your column truly stores kW (default assumes W-like values)
    """
    d = PORT_INDEX.get(port_number)
    if d is None or d.empty:
        return 0.0

    target_ts = (SIM_START_TS + pd.Timedelta(minutes=int(minute_offset))).floor("min")

    # Exact hit
    if target_ts in d.index:
        val = float(d.loc[target_ts, "p_pred_kw"])
    else:
        if not allow_nearest:
            return 0.0
        idx = d.index
        if target_ts < idx[0]:
            if not clamp_edges:
                return 0.0
            val = float(d.iloc[0]["p_pred_kw"])
        elif target_ts > idx[-1]:
            if not clamp_edges:
                return 0.0
            val = float(d.iloc[-1]["p_pred_kw"])
        else:
            pos = idx.get_indexer([target_ts], method="nearest")[0]
            if pos == -1:
                return 0.0
            val = float(d.iloc[pos]["p_pred_kw"])

    # Convert to Watts if the column is actually kW
    return val


def controller_multiple_cars(
   status, power_charging_port, storage_obj, money,
   time_step, battery_capacity, solar, wind,
   current_price, storage_capacity, grid_capacity,
   voltage, power_set_point, ev_caps, availability, charger_usage,
   delta, constant_load_time, minute_offset, controller_settings,
):
   """
   Main EMS logic for charging multiple EVs, handling energy flows, grid limits, and pricing.


   Units:
     - power: W
     - energy: Wh
     - price: $/Wh


   Sign convention for power_set_point (grid):
     + : import from grid (buy)
     - : export to grid (sell)
   """




   delta_t_h = float(delta) / 60.0


   # Treat these state codes as "charging"
   CHARGING_STATES = (1, 5)

   # Normalize ev_caps to a list of ints (Wh)
   if isinstance(ev_caps, (int, float)):
       ev_caps = [int(ev_caps)]
   else:
       ev_caps = [int(cap) for cap in ev_caps]


   # ───── EV state carry-over to next step ─────
   n_ev = len(status)
   if time_step + 1 < len(battery_capacity[0]):
       for i in range(n_ev):
           s = status[i][time_step]
           if s == 0:  # EV left hub
               battery_capacity[i][time_step + 1] = 0
           else:
               battery_capacity[i][time_step + 1] = battery_capacity[i][time_step]






   # ───── Price thresholds from YAML ─────
   with open("configurations/controller_add_config.yaml", "r") as f:
       config_data = yaml.safe_load(f)
   configs = config_data["InitializationSettings"]["configs"][0]
   price_high = float(configs["price_high"])
   price_low  = float(configs["price_low"])


   # ───── Determine who is charging this minute ─────
   charging_indices = [i for i in range(n_ev) if status[i][time_step] in CHARGING_STATES]


   # ───── Renewable power available (W) ─────
   P_total_produced = float(solar + wind)

   print(constant_load_time)

   # ───── If no EVs are charging → price-based behavior ─────
   if not charging_indices:
       # Storage capability (W) this step
       SoC_Wh  = float(storage_obj.get_soc())                   # Wh
       room_Wh = float(storage_obj.get_remaining_capacity())    # Wh
       P_dis_cap = (SoC_Wh  / delta_t_h) if delta_t_h > 0 else 0.0  # max discharge W
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
           return money, storage_obj, float(power_set_point), battery_capacity,0


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


       return money, storage_obj, float(power_set_point), battery_capacity,0


   P_const = float(constant_load_time)  # W (background/house load)

   # ───── Compute per‑EV requested power (W), limited by charger and per‑step room ─────
   P_ev_req_per_ev = []
   P_port_car1 = 0
   for i in charging_indices:
       # which physical charger is this EV using at this minute?
       charger_num = int(charger_usage[i][time_step]) if charger_usage is not None else 0

       if charger_num == 0:
           P_port = 0.0
       else:
           # 1-based to 0-based index; guard bounds
           idx = charger_num - 1
           p_kw =  get_predicted_power_W_by_minute(charger_num, minute_offset)
           P_port = float(p_kw) if p_kw is not None else 0.0

       # keep (ev_index, requested_power_W)
       P_ev_req_per_ev.append((i, P_port))
       if i == 0:
            P_port_car1 = P_port



   # total EV request (W)
   P_ev_req = sum(p for _, p in P_ev_req_per_ev)

   # ───── Storage decision (sign: +discharge to bus, -charge from bus) ─────
   SoC_Wh  = float(storage_obj.get_soc()) #Wh
   room_Wh = float(storage_obj.get_remaining_capacity()) #Wh
   P_dis_cap = (SoC_Wh / delta_t_h) if delta_t_h > 0 else 0.0
   P_chg_cap = (room_Wh / delta_t_h) if delta_t_h > 0 else 0.0
# P_dis_cap is the maximum that we can dischage the battery to
# P_chg_cap is the maximum that we can charge the battery to



#--------------if produced power ---------------------------
   if P_total_produced >= P_ev_req:
       # Surplus → charge storage
       P_surplus  = P_total_produced - P_ev_req
       P_battery_change = -min(P_surplus, P_chg_cap)  # negative means "charge the battery"
   else:
       # Deficit → discharge storage
       P_deficit  = P_ev_req - P_total_produced
       P_battery_change =  min(P_deficit, P_dis_cap)  # positive means "discharge the battery" P
#----------------battery change-------------------------------
   if P_battery_change >= 0:
       # Discharge battery; returns actual delivered W
       P_batt = storage_obj.discharge(P_battery_change, delta_t_h)
   else:
       # Charge battery; storage.charge returns +W absorbed → negate to keep bus sign
       P_batt = -storage_obj.charge(-P_battery_change, delta_t_h)








   # ───── Grid setpoint (import if storage not enough) ─────
   #If P_batt is positive it will be substructed here, if P_batt is negative, it will be added
   P_grid_unclamped = P_ev_req - P_total_produced - P_batt + P_const
   P_grid = max(-grid_capacity, min(grid_capacity, P_grid_unclamped))
   #if P_grid_unclamped = 0, P_grid = 0 (if there is no power needed)
   # if P_grid_unclamped = -1000, which is smaller than grid_capacity, P_grid = -1000 (grid_capacity = 10000), we can sell energy
   # if P_grid_unclamped = -100000, which is smaller than grid_capacity, P_grid = -10000 (grid_capacity = 10000), makes sure that it is not over grid capacity
   # if P_grid_unclamped = 1000, which is smaller than grid_capacity, P_grid = 1000 (grid_capacity = 10000), we need to buy energy





#the ratio thing makes it not the same way as it is in the matlab code
   # ───── Allocate energy to EVs proportionally if limited ─────
   P_avail_for_evs = max(0.0, P_total_produced + P_batt + P_grid - P_const)
   ratio = 1.0
   if P_ev_req > 0:
       #ratio = max(0.0, min(1.0, P_avail_for_evs / P_ev_req))
       ratio = 1


   if time_step + 1 < len(battery_capacity[0]):
       for i, P_req in P_ev_req_per_ev:
           P_alloc = P_req * ratio
           dE_Wh   = P_alloc * delta_t_h
           battery_capacity[i][time_step + 1] = min(
               float(battery_capacity[i][time_step]) + dE_Wh,
               float(ev_caps[i])
           )






   # ───── Money from grid trade ─────
   E_grid_Wh = abs(P_grid) * delta_t_h
   if P_grid > 0:
       # import (buy)
       money -= E_grid_Wh * current_price
   elif P_grid < 0:
       # export (sell)
       money += E_grid_Wh * current_price


   # Safety: ensure scalar
   if isinstance(power_set_point, list):
       power_set_point = float(power_set_point[0])


   return money, storage_obj, float(1000), battery_capacity, P_port_car1