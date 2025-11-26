import copy
import math
import random

import gym

import numpy as np
from gym import spaces
from gym.utils import seeding
from utils.Base9 import Base
from utils.HelperInitialize import HelperInitialize
from utils.InitializeMap import InitializeMap
from utils.InitializeTask import InitializeTask
from utils.UpdateCoverageBelong import UpdateCoverageBelong
from utils.LocalOffUtility import LocalOffUtility
from utils.EdgeOffUtility import EdgeOffUtility
from utils.MEC import MEC
from utils.UAV import UAV
from utils.Vehicle import Vehicle
from utils.Common1 import Common1

# 包含向本地卸载 ， 动作只映射到可以选无人机上
# actor 生成是使用字典params

class MECenv(gym.Env):
    # 初始化环境
    def __init__(self):
        random.seed(1)
        np.random.seed(1)
        self.evaluate = False
        self.info = self.init_info()
        self.n = 1
        self.agents = [self.n]
        self.seed()
        self.overboard = []
        self.max_episode_steps = 60
        self.state = []
        self.base = Base()
        self.Common1 = Common1(self.base)
        self.all_Vehicles, self.all_MECs, self.all_UAVs = HelperInitialize(self.base)
        InitializeMap(self.base, self.all_Vehicles, self.all_MECs, self.all_UAVs)
        self.all_Tasks = InitializeTask(self.base)
        self.timestep = 0
        self.users_pos = [[] for _ in range(self.base.all_Vehicles_num)]
        self.uavs_pos = [[] for _ in range(self.base.all_UAVs_num)]
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(
                                                self.base.all_MECs_num * 2 + self.base.all_UAVs_num * 2 + self.base.all_Vehicles_num * 5,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(self.base.all_UAVs_num * 2 + self.base.all_Vehicles_num * 2,),
                                       dtype=np.float32)

    def init_info(self):
        info = {}
        info['utility'] = []
        info['complete_delay'] = []
        info['proc_task_size'] = []
        info['success_num'] = []
        info['total_energy'] = []
        info['computing_energy'] = []
        info['travel_energy'] = []
        info['delay_normal'] = []
        info['energy_normal'] = []
        info['cost_normal'] = []
        info['all_info'] = []
        return info

    def update_info(self, vehicle_utility, mec_utility, complete_delay, total_energy, computing_energy, travel_energy
                    , delay_normal, energy_normal):
        self.info['utility'].append(sum(vehicle_utility) + sum(mec_utility))
        self.info['complete_delay'].append(sum(complete_delay))
        self.info['total_energy'].append(sum(total_energy) + sum(travel_energy))
        self.info['computing_energy'].append(sum(computing_energy))
        self.info['travel_energy'].append(sum(travel_energy))
        self.info['cost_normal'].append(sum(delay_normal) / self.base.all_Vehicles_num + sum(energy_normal))
        # self.info['delay_normal'].append(sum())
        proc_task_size = 0
        success_num = 0
        for i in range(self.base.all_Vehicles_num):
            if complete_delay[i] <= self.all_Tasks[i].max_delay:
                proc_task_size += self.all_Tasks[i].cpu
                success_num += 1
        self.info['proc_task_size'].append(proc_task_size)
        self.info['success_num'].append(success_num)
        self.info['delay_normal'].append(sum(delay_normal))
        self.info['energy_normal'].append(sum(energy_normal))
        if self.evaluate:
            utility = [v + m for v, m in zip(vehicle_utility, mec_utility)]
            # 创建字典
            all_info = {
                'utility': utility,
                'complete_delay': complete_delay,
                'total_energy': total_energy,
                'computing_energy': computing_energy,
                'travel_energy': travel_energy,
                'delay_normal': delay_normal,
                'energy_normal': energy_normal,
                'cost_normal': sum(delay_normal) / self.base.all_Vehicles_num + sum(energy_normal)
            }
            # 赋值给 self.info['all_info']
            self.info['all_info'].append(all_info)

    def normalize_state(self, state, min_vals, max_vals):
        if min_vals == max_vals:
            return 0
        normalized_state = (state - min_vals) / (max_vals - min_vals)
        return normalized_state

    def get_state(self):
        state = []
        # for i in range(len(self.all_MECs)):
        #     state.append(self.all_MECs[i].position[0])
        #     state.append(self.all_MECs[i].position[1])
        #     # state.append(np.array(
        #     #     self.normalize_state(self.all_MECs[i].avail_CPU, self.base.mec_cpu_freq, self.base.mec_cpu_freq)))
        # for i in range(len(self.all_UAVs)):
        #     state.append(self.all_UAVs[i].position[0])
        #     state.append(self.all_UAVs[i].position[1])
        #     # state.append(np.array(
        #     #     self.normalize_state(self.all_UAVs[i].avail_CPU, self.base.uav_cpu_freq, self.base.uav_cpu_freq)))
        # for i in range(len(self.all_Vehicles)):
        #     state.append(self.all_Vehicles[i].position[0])
        #     state.append(self.all_Vehicles[i].position[1])
        #     state.append(self.all_Vehicles[i].avail_CPU/1e8)
        #     state.append(self.all_Tasks[i].size/1e5)
        #     state.append(self.all_Tasks[i].cpu_per_bit)

        for i in range(len(self.all_MECs)):
            state.append(np.array(
                self.normalize_state(self.all_MECs[i].position[0], self.base.field_X[0], self.base.field_X[1])))
            state.append(np.array(
                self.normalize_state(self.all_MECs[i].position[1], self.base.field_Y[0], self.base.field_Y[1])))
            # state.append(np.array(
            #     self.normalize_state(self.all_MECs[i].avail_CPU, self.base.mec_cpu_freq, self.base.mec_cpu_freq)))
        for i in range(len(self.all_UAVs)):
            state.append(np.array(
                self.normalize_state(self.all_UAVs[i].position[0], self.base.field_X[0], self.base.field_X[1])))
            state.append(np.array(
                self.normalize_state(self.all_UAVs[i].position[1], self.base.field_Y[0], self.base.field_Y[1])))
            # state.append(np.array(
            #     self.normalize_state(self.all_UAVs[i].avail_CPU, self.base.uav_cpu_freq, self.base.uav_cpu_freq)))
        for i in range(len(self.all_Vehicles)):
            state.append(np.array(
                self.normalize_state(self.all_Vehicles[i].position[0], self.base.field_X[0], self.base.field_X[1])))
            state.append(np.array(
                self.normalize_state(self.all_Vehicles[i].position[1], self.base.field_Y[0], self.base.field_Y[1])))
            state.append(np.array(self.normalize_state(self.all_Vehicles[i].avail_CPU, 0.5e9, 0.6e9)))
            state.append(np.array(self.normalize_state(self.all_Tasks[i].size, 1e6, 5e6)))
            state.append(np.array(self.normalize_state(self.all_Tasks[i].cpu_per_bit, 500, 1500)))
        return state

    def seed(self, seed=None):
        self.seed_value = seed
        np.random.seed(seed)
    def set_evaluate(self):
        self.evaluate = True
    # 重置环境，输出重置后的状态,包括无人机(位置，CPU),MEC(位置，CPU), 任务(位置，大小，cpu_per_bit，时延约束)
    def reset(self):
        self.evaluate = False
        self.timestep = 0
        self.state = []
        self.all_Vehicles, self.all_MECs, self.all_UAVs = HelperInitialize(self.base)
        InitializeMap(self.base, self.all_Vehicles, self.all_MECs, self.all_UAVs)
        self.all_Tasks = InitializeTask(self.base)
        self.Common1.UpdateVehiclePosition(self.all_Vehicles, self.timestep)
        self.state = self.get_state()
        # for task in self.all_Tasks:
        #     print("Vehicle position:", task.cpu,task.size,task.cpu_per_bit,task.max_delay)
        self.users_pos = [[] for _ in range(len(self.all_Vehicles))]
        self.uavs_pos = [[] for _ in range(len(self.all_UAVs))]
        for i in range(len(self.all_Vehicles)):
            self.users_pos[i].append(np.array(self.all_Vehicles[i].position[:2]))
        for i in range(len(self.all_UAVs)):
            self.uavs_pos[i].append(np.array(self.all_UAVs[i].position[:2]))
        self.overboard = []
        self.info = self.init_info()
        return np.array(self.state)

    # 外部接口函数，输入为动作，输出下一个状态，奖励，智能体是否终止以及其他信息
    def step(self, actions):
        # print(actions)
        done = False
        self.timestep += 1
        if self.timestep >= self.max_episode_steps:
            done = True
        utility_overboard = 0.
        uavs_velocity = []
        uavs_position = []
        for i in range(self.base.all_UAVs_num):
            d_x = (actions[2 * i]) * 30
            d_y = (actions[2 * i + 1]) * 30
            new_x = self.all_UAVs[i].position[0] + d_x
            new_y = self.all_UAVs[i].position[1] + d_y
            out_of_bounds = False
            if new_x > self.base.field_X[1]:
                new_x = self.base.field_X[1]
                out_of_bounds = True
            elif new_x < self.base.field_X[0]:
                new_x = self.base.field_X[0]
                out_of_bounds = True
            if new_y > self.base.field_Y[1]:
                new_y = self.base.field_X[1]
                out_of_bounds = True
            elif new_y < self.base.field_Y[0]:
                new_y = self.base.field_Y[0]
                out_of_bounds = True
            if out_of_bounds:
                utility_overboard += -50
            # Update the position
            uavs_velocity.append(np.sqrt(d_x ** 2 + d_y ** 2) / self.base.run_slot)
            uavs_position.append(np.array([new_x, new_y, self.base.UAV_altitude]))

        self.Common1.UpdateUAVPosition(self.all_UAVs, uavs_velocity, np.array(uavs_position), self.timestep)
        #  link:连接矩阵（前MEC后UAV）
        link, distance, offload_rate, connect_num = UpdateCoverageBelong(self.base, self.all_Vehicles, self.all_MECs,
                                                                         self.all_UAVs)
        #  off存放任务的车辆id，res存放分配的资源大小
        offload_policy_list = [[] for _ in range(self.base.all_MECs_num + self.base.all_UAVs_num + 1)]
        resource_allocation_list = [[] for _ in range(self.base.all_MECs_num + self.base.all_UAVs_num + 1)]
        price_policy = [self.base.default_increase_price for _ in range(self.base.all_Vehicles_num)]
        link_num = [0 for _ in range(self.base.all_MECs_num + self.base.all_UAVs_num)]  # 已连接数量，

        #  offload
        for i in range(self.base.all_Vehicles_num):
            offload_policy = (actions[2 * self.base.all_UAVs_num + 2 * i]) / 2 + 0.5
            resource_allocation = (actions[2 * self.base.all_UAVs_num + 2 * i + 1]) / 2 + 0.5
            policy_range = 1.0 / (sum(link[i]) + 1)
            index = min(int(offload_policy // policy_range), sum(link[i]))
            if index == sum(link[i]):
                offload_policy_list[-1].append(i)
                resource_allocation_list[-1].append(resource_allocation)
            else:
                count = 0
                for j, val in enumerate(link[i]):
                    if val == 1:
                        count += 1
                        if count == index + 1:
                            offload_policy_list[j].append(i)
                            resource_allocation_list[j].append(resource_allocation)
                            break
            # index_distance = [(index + 1, dist) for index, dist in enumerate(distance[i][1:])]  # 全部uav距离，下标从1开始
            # sorted_pairs = sorted(index_distance, key=lambda x: x[1])
            # nearest_index = 0
            # for pair in sorted_pairs:
            #     if link_num[pair[0]] <= self.base.max_link_num and link[i][pair[0]] == 1:
            #         nearest_index = pair[0]
            #         link_num[pair[0]] += 1
            # offload_policy_list[nearest_index].append(i)
            # resource_allocation_list[nearest_index].append(resource_allocation)
        # #  就近卸载（不存在最大连接数量）
        # for i in range(self.base.all_Vehicles_num):
        #
        #     offload_policy_index = np.argmin(distance[i][1:]) + 1
        #     resource_allocation = (actions[2 * self.base.all_UAVs_num + 2 * i]) / 2 + 0.5
        #     price_policy[i] = 1e-11+(actions[2 * self.base.all_UAVs_num + 2 * i + 1]+1)*(2e-8 - 1e-11)/2
        #     if link[i][offload_policy_index] ==1:
        #         offload_policy_list[offload_policy_index].append(i)
        #         resource_allocation_list[offload_policy_index].append(resource_allocation)
        #     else:
        #         offload_policy_list[0].append(i)
        #         resource_allocation_list[0].append(resource_allocation)

        # 存放真实分配的资源量
        final_allocation_list = [0 for _ in range(self.base.all_Vehicles_num)]
        # 存放真实上传速率
        updated_rate_list = [0 for _ in range(self.base.all_Vehicles_num)]
        # 存放连接的服务器id
        offload_severs_list = [-1 for _ in range(self.base.all_Vehicles_num)]
        for i in range(self.base.all_MECs_num + self.base.all_UAVs_num + 1):
            if not offload_policy_list[i]:
                continue
            connect_num = len(offload_policy_list[i])
            for j in range(len(offload_policy_list[i])):
                if i < self.base.all_MECs_num:
                    resource = resource_allocation_list[i][j] * self.all_MECs[
                        i].avail_CPU / self.base.all_Vehicles_num * 2 + self.all_Vehicles[
                                   offload_policy_list[i][j]].avail_CPU
                    offload_rate = self.Common1.CalcUpDataRateUE2BS(self.all_Vehicles[offload_policy_list[i][j]],
                                                                    self.all_MECs[i], connect_num)
                elif i < self.base.all_UAVs_num + self.base.all_MECs_num:
                    resource = resource_allocation_list[i][j] * self.all_UAVs[
                        i - self.base.all_MECs_num].avail_CPU / self.base.all_Vehicles_num * 2 + self.all_Vehicles[
                                   offload_policy_list[i][j]].avail_CPU
                    offload_rate = self.Common1.CalcUpDataRateUE2UAV(self.all_Vehicles[offload_policy_list[i][j]],
                                                                     self.all_UAVs[i - self.base.all_MECs_num],
                                                                     connect_num)
                else:
                    resource = -1
                    offload_rate = -1
                final_allocation_list[offload_policy_list[i][j]] = resource
                updated_rate_list[offload_policy_list[i][j]] = offload_rate
                offload_severs_list[offload_policy_list[i][j]] = i
        # 计算奖励\
        utility = 0
        vehicle_utility = [0 for _ in range(len(self.all_Vehicles))]
        mec_utility = [0 for _ in range(len(self.all_Vehicles))]
        complete_delay = [0 for _ in range(len(self.all_Vehicles))]
        upload_delay = [0 for _ in range(len(self.all_Vehicles))]
        total_energy = [0 for _ in range(len(self.all_Vehicles))]
        upload_energy = [0 for _ in range(len(self.all_Vehicles))]
        computation_energy = [0 for _ in range(len(self.all_Vehicles))]
        complete_delay_normal = [0 for _ in range(len(self.all_Vehicles))]
        energy_normal = [0 for _ in range(len(self.all_Vehicles))]
        travel_energy = []
        for dest_server in self.all_UAVs:
            cur_travel_energy = (self.base.uav_energy_par1 * (
                    1 + 3 * np.linalg.norm(dest_server.velocity) ** 2 / dest_server.tip_speed_rotor_blade ** 2)
                                 + self.base.uav_energy_par2 * np.sqrt(
                        np.sqrt(
                            self.base.uav_energy_par3 + np.linalg.norm(dest_server.velocity) ** 4 / 4) - np.linalg.norm(
                            dest_server.velocity) ** 2 / 2)
                                 + self.base.uav_energy_par4 * np.linalg.norm(
                        dest_server.velocity) ** 3) * self.base.run_slot
            travel_energy.append(cur_travel_energy)
        for i in range(len(self.all_Vehicles)):
            if offload_severs_list[i] == self.base.all_MECs_num + self.base.all_UAVs_num:
                (vehicle_utility[i], complete_delay[i], total_energy[i], complete_delay_normal[i], energy_normal[i]
                 ) = (LocalOffUtility(self.base, self.all_Vehicles[i], self.all_Tasks[i],
                                                    self.all_Vehicles[i].avail_CPU,price_policy[i]))
                computation_energy[i] = total_energy[i]
            else:
                if offload_severs_list[i] < self.base.all_MECs_num:
                    off_load_server = self.all_MECs[offload_severs_list[i]]
                else:
                    off_load_server = self.all_UAVs[offload_severs_list[i] - self.base.all_MECs_num]
                (vehicle_utility[i], mec_utility[i], complete_delay[i], upload_delay[i], computation_energy[i],
                 upload_energy[i], complete_delay_normal[i], energy_normal[i]) = EdgeOffUtility(self.base,
                                                                                                self.all_Vehicles[i],
                                                                                                self.all_Tasks[i],
                                                                                                off_load_server,
                                                                                                final_allocation_list[
                                                                                                    i], price_policy[i],
                                                                                                updated_rate_list[i])
                (temp_vehicle_utility, temp_complete_delay, temp_total_energy, temp_complete_delay_normal,
                 temp_total_energy_normal) = LocalOffUtility(self.base, self.all_Vehicles[i],
                                                                                         self.all_Tasks[i],
                                                                                         self.all_Vehicles[i].avail_CPU,
                                                                                         price_policy[i])
                if temp_complete_delay < complete_delay[i] and self.evaluate:
                    complete_delay[i] = temp_complete_delay
                    vehicle_utility[i] = temp_vehicle_utility
                    mec_utility[i] = 0
                    upload_delay[i] = 0
                    computation_energy[i] = temp_total_energy
                    upload_energy[i] = 0
                    complete_delay_normal[i] = temp_complete_delay_normal
                    energy_normal[i] = temp_total_energy_normal
                total_energy[i] = computation_energy[i] + upload_energy[i]
        utility += sum(mec_utility) + sum(vehicle_utility)
        # 传输info
        self.update_info(vehicle_utility, mec_utility, complete_delay, total_energy, computation_energy, travel_energy
                         , complete_delay_normal, energy_normal)
        # 记录轨迹，画图用
        for i in range(len(self.all_Vehicles)):
            self.users_pos[i].append(np.array(self.all_Vehicles[i].position[:2]))
        for i in range(len(self.all_UAVs)):
            self.uavs_pos[i].append(np.array(self.all_UAVs[i].position[:2]))
        # 更新任务和用户位置
        self.Common1.UpdateVehiclePosition(self.all_Vehicles, self.timestep)
        self.all_Tasks = InitializeTask(self.base)
        # 获取next_state
        next_state = self.get_state()
        reward = -sum(complete_delay) * 1 + -(
                sum(total_energy) + sum(travel_energy)) * 0.3 + utility_overboard
        # reward = -sum(complete_delay) * 0.1 + -(
        #         sum(computation_energy) + sum(travel_energy)) * 0.0003 + utility_overboard
        # print(f"reward = {reward}, overboard = {utility_overboard}")
        self.overboard.append(utility_overboard)
        info = {}
        info['info'] = copy.deepcopy(self.info)
        info['utility'] = utility  # 总效用
        info['vehicle_utility'] = sum(vehicle_utility) / self.base.all_Vehicles_num  # 车辆效用
        info['mec_utility'] = sum(mec_utility) / (self.base.all_MECs_num + self.base.all_UAVs_num)  # mec节点效用
        info['complete_delay'] = sum(complete_delay)  # 总时延
        info['upload_delay'] = sum(upload_delay)  # 上传时延
        info['total_energy'] = sum(total_energy)  # 总能耗（不包括飞行）
        info['computation_energy'] = sum(computation_energy)  # 计算能耗
        info['travel_energy'] = sum(travel_energy)  # 飞行能耗
        info['delay_normal'] = sum(complete_delay_normal)
        info['energy_normal'] = sum(energy_normal)
        info['cost_normal'] = sum(complete_delay_normal) / self.base.all_Vehicles_num + sum(energy_normal)
        info['out_of_bound'] = utility_overboard
        return np.array(next_state), reward, done, info

    def render(self):
        # 传输用户位置以及无人机的全部位置
        user_positions = copy.deepcopy(self.users_pos)
        uav_positions = copy.deepcopy(self.uavs_pos)
        return user_positions, uav_positions

    def get_overboard(self):
        return np.mean(self.overboard)
# env = MECenv()
# state = env.reset()
# link, distance, up_data_rate, connect_num = UpdateCoverageBelong(env.all_Vehicles, env.all_MECs, env.all_UAVs)
#
# for i in range(5):
#     (state, reward, done, info) = env.step(np.random.uniform(-1, 1, 2 * env.base.all_UAVs_num + 2 * env.base.all_Vehicles_num))
