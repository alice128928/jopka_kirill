# BatteryStorage class
# --------------------
# This class models a simple battery storage system with charge/discharge efficiency and state-of-charge (SoC) tracking.
#
# Inputs:
# - capacity_kWh (float): Total energy capacity of the battery in kilowatt-hours (Wh)
# - charge_eff (float): Charging efficiency (0 < charge_eff ≤ 1). Default is 0.95.
# - discharge_eff (float): Discharging efficiency (0 < discharge_eff ≤ 1). Default is 0.95.
# - initial_soc (float): Initial state of charge in Wh. Default is 0.0 (empty).
#
# Outputs (from methods):
# - charge(): Actual power consumed (in W) to charge the battery over the time step
# - discharge(): Actual power delivered (in W) by the battery over the time step
# - get_soc(): Current energy stored in the battery (in Wh)
# - get_remaining_capacity(): Unused energy capacity (in Wh)

class BatteryStorage:
    def __init__(self, capacity_Wh, charge_eff=0.95, discharge_eff=0.95, initial_soc=0.0):
        """
        Initialize a BatteryStorage object.

        Parameters:
        - capacity_Wh (float): Total energy capacity of the battery in Wh.
        - charge_eff (float): Efficiency of charging (default 0.95).
        - discharge_eff (float): Efficiency of discharging (default 0.95).
        - initial_soc (float): Initial state of charge in Wh (default 0.0).
        """
        self.capacity = capacity_Wh
        self.charge_eff = charge_eff
        self.discharge_eff = discharge_eff
        self.SoC = initial_soc  # Current energy stored in Wh

    def charge(self, power_W, delta_t_h):
        """
        Charge the battery for a given time with a specified power input.

        Parameters:
        - power_W (float): Input power in kilowatts.
        - delta_t_h (float): Duration of charging in hours.

        Returns:
        - float: Actual power (in kW) drawn from the source during this period,
                 accounting for charge efficiency and capacity limits.
        """
        energy_added = power_W * delta_t_h * self.charge_eff  # energy added after efficiency losses
        actual_added = min(self.capacity - self.SoC, energy_added)  # limit to max capacity
        self.SoC += actual_added  # update state of charge
        return actual_added / (self.charge_eff * delta_t_h)  # return real input power drawn

    def discharge(self, power_W, delta_t_h):
        """
        Discharge the battery for a given time at a specified power level.

        Parameters:
        - power_W (float): Output power requested in kilowatts.
        - delta_t_h (float): Duration of discharging in hours.

        Returns:
        - float: Actual power (in W) delivered to the load during this period,
                 accounting for discharge efficiency and available energy.
        """
        energy_needed = power_W * delta_t_h / self.discharge_eff  # required energy including losses
        actual_drawn = min(self.SoC, energy_needed)  # limit to what is stored
        self.SoC -= actual_drawn  # update state of charge
        return actual_drawn * self.discharge_eff / delta_t_h  # return real output power delivered

    def get_soc(self):
        """
        Get the current state of charge (SoC) of the battery.

        Returns:
        - float: Energy currently stored in Wh.
        """
        return self.SoC

    def get_soc_percentage(self):
        """
        Get the current state of charge (SoC) of the battery.

        Returns:
        - percentage
        """
        a = (self.SoC/ self.capacity)*100
        return a



    def get_remaining_capacity(self):
        """
        Get the remaining unused capacity of the battery.

        Returns:
        - float: Available capacity in kWh.
        """
        return self.capacity - self.SoC



