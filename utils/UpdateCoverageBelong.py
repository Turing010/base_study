import numpy as np
from utils.Common1 import Common1

def UpdateCoverageBelong(Base, all_Vehicles, all_MECs, all_UAVs):
    base = Base
    common1 = Common1(base)

    link = np.zeros((base.all_Vehicles_num, base.all_MECs_num + base.all_UAVs_num))
    distance = 10 ** 5 * np.ones((base.all_Vehicles_num, base.all_MECs_num + base.all_UAVs_num))
    up_data_rate = np.zeros((base.all_Vehicles_num, base.all_MECs_num + base.all_UAVs_num))
    connect_num = np.zeros(base.all_MECs_num + base.all_UAVs_num)

    for j in range(base.all_MECs_num + base.all_UAVs_num):
        connect_num[j] = 0
        for i in range(base.all_Vehicles_num):
            cur_veh = all_Vehicles[i]
            if 1 <= j + 1 <= base.all_MECs_num:
                cur_mec = all_MECs[j]
                distance[i, j] = np.linalg.norm(np.array(cur_mec.position[:2]) - np.array(cur_veh.position[:2]))
                if distance[i, j] <= cur_mec.coverage_radius:
                    link[i, j] = 1
                    connect_num[j] += 1
            else:
                uav_index = j - base.all_MECs_num
                cur_mec = all_UAVs[uav_index]
                distance[i, j] = np.linalg.norm(np.array(cur_mec.position[:2]) - np.array(cur_veh.position[:2]))
                if distance[i, j] <= cur_mec.coverage_radius:
                    link[i, j] = 1
                    connect_num[j] += 1

    for i in range(base.all_Vehicles_num):
        cur_veh = all_Vehicles[i]
        for j in range(base.all_MECs_num + base.all_UAVs_num):
            if 1 <= j + 1 <= base.all_MECs_num:
                if link[i, j] == 1:
                    cur_mec = all_MECs[j]
                    up_data_rate[i, j] = common1.CalcUpDataRateUE2BS(cur_veh, cur_mec, 1)
            else:
                uav_index = j - base.all_MECs_num
                cur_mec = all_UAVs[uav_index]
                if link[i, j] == 1:
                    link[i, j] = 1
                    up_data_rate[i, j] = common1.CalcUpDataRateUE2UAV(cur_veh, cur_mec, 1)

    return link, distance, up_data_rate, connect_num

