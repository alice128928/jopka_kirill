import yaml
from battery_storage import BatteryStorage  # keep import if your project expects it


def controller_multiple_cars(
   status, power_charging_port, storage_obj, money,
   time_step, battery_capacity, solar, wind,
   current_price, storage_capacity, grid_capacity,
   voltage, power_set_point, ev_caps, availability, charger_usage,
   delta, constant_load_time, controller_settings,
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
   CHARGING_STATES = 1

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
   charging_indices = [i for i in range(n_ev) if status[i][time_step] == CHARGING_STATES]


   # ───── Renewable power available (W) ─────
   P_total_produced = float(solar + wind)


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


       return money, storage_obj, float(power_set_point), battery_capacity, 0


   P_const = float(constant_load_time)  # W (background/house load)

   # ───── Compute per‑EV requested power (W), limited by charger and per‑step room ─────
   P_ev_req_per_ev = []

   for i in charging_indices:
       # which physical charger is this EV using at this minute?
       charger_num = int(charger_usage[i][time_step]) if charger_usage is not None else 0

       if charger_num == 0:
           P_port = 0.0
       else:
           # 1-based to 0-based index; guard bounds
           idx = charger_num - 1
           P_port = float(power_charging_port[idx])

       # keep (ev_index, requested_power_W)
       P_ev_req_per_ev.append((i, P_port))
       #if i == 0:
           #print("EV 0 requests:", P_ev_req_per_ev)




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


   return money, storage_obj, float(1000), battery_capacity, P_port