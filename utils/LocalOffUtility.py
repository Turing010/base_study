import numpy as np

def LocalOffUtility(base, vehicle, task, str_allo, str_price):
    vehicle_utility = 0
    complete_delay = 0
    compute_energy = 0
    complete_delay_normal = 0
    compute_energy_normal = 0
    cost_normal = 0
    if str_allo > 0:
        complete_delay = task.cpu / str_allo
        complete_delay_normal = complete_delay / base.task_max_delay
        compute_energy = vehicle.CPU_chip_params[0] * (str_allo ** (vehicle.CPU_chip_params[1] - 1)) * task.cpu
        compute_energy_normal = compute_energy / vehicle.energy_constraint
        satis_level_normal = np.log(1 + max(0, task.max_delay - complete_delay)) / np.log(1 + task.max_delay)
        vehicle_utility = vehicle.weight_factor * satis_level_normal \
            - (1 - vehicle.weight_factor) * (compute_energy / vehicle.energy_constraint
                                                + str_price / vehicle.max_budget)

        if vehicle_utility < 0:
            vehicle_utility = 0

    return vehicle_utility, complete_delay, compute_energy, complete_delay_normal, compute_energy_normal
