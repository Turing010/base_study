import numpy as np


def EdgeOffUtility(base, vehicle, task, dest_server, str_allo, str_price, up_data_rate):
    if up_data_rate == -1:
        print("updaterate==-1!")
    veh_utility = 0
    mec_utility = 0
    complete_delay = 0
    total_energy = 0
    travel_energy = 0
    computation_energy = 0
    complete_delay_normal = 0
    energy_normal = 0
    cost_normal = 0
    # Communication
    upload_delay = task.size / up_data_rate + 194 * 8 / up_data_rate
    upload_energy = vehicle.trans_power_w * upload_delay

    # Computation
    computation_delay = task.cpu / str_allo if str_allo > 0 else 0

    # Total delay
    complete_delay = upload_delay + computation_delay

    if str_allo <= 0:
        veh_utility = 0
        mec_utility = 0
    else:
        # Vehicle utility
        satis_level_normal = np.log(1 + max(0, task.max_delay - complete_delay)) / np.log(1 + task.max_delay)
        veh_utility = vehicle.weight_factor * satis_level_normal - (1 - vehicle.weight_factor) * (
                    upload_energy / vehicle.energy_constraint
                    + (str_allo * str_price) / vehicle.max_budget)

        veh_utility = max(0, veh_utility)

        computation_energy = dest_server.CPU_chip_params[0] * (
                str_allo ** (dest_server.CPU_chip_params[1] - 1)) * task.cpu
        up_comp_energy = upload_energy + computation_energy

        # 归一化时延能耗
        complete_delay_normal = complete_delay / base.task_max_delay
        energy_normal = upload_energy / vehicle.energy_constraint + computation_energy / dest_server.energy_constraint
        cost_normal = complete_delay_normal + energy_normal
        if dest_server.type == 'MEC':
            mec_utility = dest_server.weight_factor * str_allo * str_price / (
                    dest_server.max_unit_pricing * dest_server.CPU_freq) \
                          - (1 - dest_server.weight_factor) * (computation_energy / dest_server.energy_constraint)

            mec_utility = max(0, mec_utility)

            total_energy = upload_energy + computation_energy

        elif dest_server.type == 'UAV':
            travel_energy = (base.uav_energy_par1 * (
                    1 + 3 * np.linalg.norm(dest_server.velocity) ** 2 / dest_server.tip_speed_rotor_blade ** 2)
                             + base.uav_energy_par2 * np.sqrt(
                        np.sqrt(base.uav_energy_par3 + np.linalg.norm(dest_server.velocity) ** 4 / 4) - np.linalg.norm(
                            dest_server.velocity) ** 2 / 2)
                             + base.uav_energy_par4 * np.linalg.norm(dest_server.velocity) ** 3) * base.run_slot
            # total_energy = upload_energy + computation_energy

            mec_utility = dest_server.weight_factor * str_allo * str_price / (
                    dest_server.max_unit_pricing * dest_server.CPU_freq) \
                          - (1 - dest_server.weight_factor) * (
                                  (computation_energy + travel_energy) / dest_server.energy_constraint)

            mec_utility = max(0, mec_utility)

        else:
            raise ValueError('No such mec type: {}'.format(dest_server.type))

    return (veh_utility, mec_utility, complete_delay, upload_delay, computation_energy, upload_energy
            , complete_delay_normal, energy_normal)
