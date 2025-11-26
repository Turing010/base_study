import numpy as np
import random
from scipy.stats import nakagami
import math
class Common1:
    def __init__(self, base):
        self.base = base

    ## nearest MEC
    def NearstServer(self, veh, distance, link, all_MECs, all_UAVs):
        i = veh.id

        if distance[i].size > 0:
            min_dis, j = np.min(distance[i]), np.argmin(distance[i])

            if link[i, j] > 0:
                if 1 <= j <= self.base.all_MECs_num:
                    mec_index = j
                    dest_server = all_MECs[mec_index]
                elif self.base.all_MECs_num < j <= self.base.all_MECs_num + self.base.all_UAVs_num:
                    uav_index = j - self.base.all_MECs_num
                    dest_server = all_UAVs[uav_index]
            else:
                dest_server = veh
                print('No nearest server!')
        else:
            dest_server = veh
            print('No nearest server!')

        return dest_server

    ## Average Allocation for Nearest vehicles
    def AverageNearAllocation(self, dest_server, distance, link, all_Vehicles, all_MECs, all_UAVs):
        near_num = 0

        for i in range(self.base.all_Vehicles_num):
            cur_veh = all_Vehicles[i]
            near_server = self.NearstServer(cur_veh, distance, link, all_MECs, all_UAVs)

            if near_server.id == dest_server.id:
                near_num += 1

        if dest_server.queue_length > 0:
            cur_str_allo = dest_server.avail_CPU / min(near_num, dest_server.queue_length)
        else:
            cur_str_allo = 0

        return cur_str_allo

    def AverageAllocation1(self, dest_server):
        if dest_server.queue_length > 0:
            cur_str_allo = dest_server.avail_CPU / max(15, dest_server.queue_length)
        else:
            cur_str_allo = 0

        return cur_str_allo

    def AverageAllocation(self, dest_server):
        if dest_server.queue_length > 0:
            cur_str_allo = dest_server.avail_CPU / dest_server.queue_length
        else:
            cur_str_allo = 0

        return cur_str_allo

    ## calculate upload data rate from UE to BS
    def CalcUpDataRateUE2BS(self, sender, receiver, connect_num):
        distance = np.linalg.norm(np.array(sender.position) - np.array(receiver.position))
        prob_LoS = min(self.base.d1 / distance, 1) * (1 - np.exp(- distance / self.base.d2)) + np.exp(- distance / self.base.d2)
        channel_power_gain_LoS = self.CalcChannelPowerGainUE2BS(sender, receiver, self.base.path_loss_UE_LoS, self.base.nakagami_UE_LoS, self.base.shadow_UE_LoS)
        channel_power_gain_NLoS = self.CalcChannelPowerGainUE2BS(sender, receiver, self.base.path_loss_UE_LoS, self.base.nakagami_UE_NLoS, self.base.shadow_UE_NLoS)
        channel_power_gain = prob_LoS * channel_power_gain_LoS + (1 - prob_LoS) * channel_power_gain_NLoS
        upload_rate = receiver.max_bandwidth / connect_num * np.log2(1 + sender.trans_power_w * channel_power_gain / self.base.noise_power)
        return upload_rate

    ## calculate channel power gain from UE to BS
    import numpy as np
    from scipy.stats import nakagami

    def CalcChannelPowerGainUE2BS(self, sender, receiver, path_loss_exponent, nakagami_m, shadow_variance):
        distance = np.linalg.norm(sender.position - receiver.position)
        # path loss
        pathloss_ref = (4 * np.pi * self.base.reference_distance * self.base.carrier_frequency / self.base.light_speed) ** 2
        pathloss_d = (distance / self.base.reference_distance) ** path_loss_exponent
        pathloss = pathloss_ref * pathloss_d
        # shadowing
        shadowing = 10 ** (shadow_variance * np.random.randn(1, 1) / 10)[0][0]  # in dB
        # fast fading
        pd_nakagami = nakagami(nakagami_m, scale=self.base.nakagami_aver_gain)
        fast_fading_nakagami = np.mean(pd_nakagami.rvs(size=self.base.fading_num) ** 2)
        channel_power_gain = fast_fading_nakagami / (pathloss * shadowing)
        # print('UE', sender.id, 'to BS', receiver.id, ':', 10 * np.log10(channel_power_gain), "dB")
        return channel_power_gain

    def CalcUpDataRateUE2UAV(self, sender, receiver, connect_num):
        # Calculate distance between sender and receiver
        distance = np.linalg.norm(np.array(sender.position) - np.array(receiver.position))

        # Calculate probability of LoS transmission
        prob_LoS = 1 / (1 + self.base.UAV_LoS_a * np.exp(
            -self.base.UAV_LoS_b * (180 / np.pi * math.asin(receiver.position[2] / distance) - self.base.UAV_LoS_a)))

        # Calculate channel power gain for LoS and NLoS
        channel_power_gain_LoS = self.CalcChannelPowerGainUE2UAV(distance, self.base.path_loss_UAV,
                                                            self.base.nakagami_UAV_LoS, 1)
        channel_power_gain_NLoS = self.CalcChannelPowerGainUE2UAV(distance, self.base.path_loss_UAV,
                                                             self.base.nakagami_UAV_NLoS, self.base.addit_atten_UAV_NLoS)

        # Combine LoS and NLoS channel power gain
        channel_power_gain = prob_LoS * channel_power_gain_LoS + (1 - prob_LoS) * channel_power_gain_NLoS

        # Calculate upload rate
        upload_rate = receiver.max_bandwidth / connect_num * np.log2(
            1 + sender.trans_power_w * channel_power_gain / self.base.noise_power)

        return upload_rate

    def CalcChannelPowerGainUE2UAV(self, distance, path_loss_exponent, nakagami_m, shadowing):
        # path loss
        pathloss_ref = (
                               4 * np.pi * self.base.reference_distance * self.base.carrier_frequency / self.base.light_speed) ** 2
        pathloss_d = (distance / self.base.reference_distance) ** path_loss_exponent
        pathloss = pathloss_ref * pathloss_d
        # fast fading
        pd_nakagami = nakagami(nakagami_m, scale=self.base.nakagami_aver_gain)
        fast_fading_nakagami = np.mean(pd_nakagami.rvs(size=self.base.fading_num) ** 2)
        channel_power_gain = fast_fading_nakagami / (pathloss * shadowing)
        # print('UE', sender.id, 'to UAV', receiver.id, ':', 10 * np.log10(channel_power_gain), "dB")
        return channel_power_gain

    ## Minimum required CPU
    def CalcMinimumRequiredCPU(self, task, up_data_rate):
        vehicle_upload_delay = task.size / up_data_rate
        req_cpu_lower_bound = task.cpu / (task.max_delay - self.base.min_satisfacion_level - vehicle_upload_delay)
        return req_cpu_lower_bound

    ## Markov_Random
    def MarkovRandom(self, pre_var, memory_level, mean_var, Gauss_variance):
        Gauss_process = np.random.normal(0, Gauss_variance)
        cur_var = memory_level * pre_var + (1 - memory_level) * mean_var + np.sqrt(1 - memory_level ** 2) * Gauss_process
        return cur_var

    ## Update Vehicle Position
    def UpdateVehiclePosition(self, all_Vehicles, cur_time):
        #print('---------------Position Update for UE-----------------------------------------')

        for i in range(len(all_Vehicles)):
            vehicle = all_Vehicles[i]

            if cur_time == 1:
                vehicle.UpdateTrajectory(cur_time, vehicle.position, vehicle.velocity, vehicle.direction)
            else:
                cur_velocity = self.MarkovRandom(vehicle.velocity, self.base.veh_memory_level_velocity,
                                                 self.base.veh_mean_velocity, self.base.veh_Gauss_variance_velocity)
                cur_direction = self.MarkovRandom(vehicle.direction, self.base.veh_memory_level_direction,
                                                  self.base.veh_mean_direction, self.base.veh_Gauss_variance_direction)
                cur_position = [
                    vehicle.position[0] + vehicle.velocity * math.cos(vehicle.direction) * self.base.mobility_slot,
                    vehicle.position[1] + vehicle.velocity * math.sin(vehicle.direction) * self.base.mobility_slot,
                    0]
                vehicle.UpdateVelocity(cur_velocity)
                vehicle.UpdateDirection(cur_direction)
                vehicle.UpdatePosition(cur_position)
                vehicle.UpdateTrajectory(cur_time, vehicle.position, vehicle.velocity, vehicle.direction)
                # print('Vehicle id = {}, velocity = {}, direction = {}, position = ({}, {})'.format(vehicle.id, vehicle.velocity, vehicle.direction, vehicle.position[0], vehicle.position[1]))

    ## Update UAV Position
    def UpdateUAVPosition(self, all_UAVs, uav_current_velocity, uav_current_pos, cur_time):
        for i in range(len(all_UAVs)):
            all_UAVs[i].UpdateVelocity(uav_current_velocity[i])
            all_UAVs[i].UpdatePosition(uav_current_pos[i])
            all_UAVs[i].UpdateTrajectory(cur_time, all_UAVs[i].velocity, all_UAVs[i].position)

    # %%sojourn time
    # % trajectory = struct('time', {}, 'pos', {}, 'vel', {});
    # %min_time, min_dis消除我们希望得到的两个时间点的误差 cur_time, pre_time
    # %没有给定pre_time为了后面获取不同time - scale信息
    # % vehicle_move_indicator = 1 towards; -1 away; 0
    def CalSojournTimeSimple(vehicle, mec):
        size_trajectory = len(vehicle.trajectory)
        vehicle_cur_position = vehicle.position
        vehicle_cur_velocity = vehicle.velocity

        if size_trajectory > 1:
            vehicle_cur_position = vehicle.trajectory[-1]['pos']
            vehicle_cur_velocity = vehicle.trajectory[-1]['vel']

            cur_distance = abs(vehicle_cur_position[0] - mec.position[0])
            initial_distance = abs(vehicle.trajectory[0]['pos'][0] - mec.position[0])

            if cur_distance > initial_distance:
                vehicle_move_indicator = -1
                print(f'Vehicle {vehicle.id} moves away from MEC {mec.id}')
            elif cur_distance < initial_distance:
                vehicle_move_indicator = 1
                print(f'Vehicle {vehicle.id} moves towards MEC {mec.id}')
            else:
                vehicle_move_indicator = 0
                print(f'Vehicle {vehicle.id} keeps stationary in the range of MEC server {mec.id}')

        else:
            print('Only one element or no element in the trajectory set. It seems that it is the initial time slot')
            pd_markov = np.random.normal(0.5, 1)
            vehicle_move_indicator = pd_markov

        sojourn_time = (mec.coverage_radius + vehicle_move_indicator * abs(
            vehicle_cur_position[0] - mec.position[0])) / abs(vehicle_cur_velocity[0])
        return sojourn_time