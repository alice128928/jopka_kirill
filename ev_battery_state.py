"""
    Checks charging availability status for each car at a given time_step.

    Parameters:
    - time_step: integer representing the current time step (hour)
    - avail: list of numpy arrays, each representing availability status (2 = away, 3 = at charger)
    - batteries: list of maximum battery capacities for each car
    - current_battery: list of numpy arrays representing current battery charge per hour for each car

    Returns:
    - status :
        0 = not charging, means car is away and is not at the charging port
        1 = can charge, there is a battery capacity available for the car to continue charging
        5 = not charging, there is no battery capacity available for the car to continue charging
        status is an array, if there is one car it will return 1, 0 or 5. if there are more cars it will return
        an array with 0s,1s and 5s depending on the status of each car.
    """
import numpy as np

def ev_state(time_step, avail, batteries, current_battery):
    status = []
    for i in range(len(avail)):
        car_status = avail[i][time_step]
        if car_status != 3:            # not at charger
            status.append(0)
            continue

        cap = float(batteries[i])
        soc = float(current_battery[i][time_step])

        # consider "full" if within 0.1% or within 1 Wh (tune as needed)
        is_full = (soc >= cap) or np.isclose(soc, cap, rtol=1e-3, atol=1.0)

        if is_full:
            status.append(5)           # at charger, already full
        else:
            status.append(1)           # at charger, can still charge
    return status
