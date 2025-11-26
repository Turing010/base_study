from utils.Actor import Actor
import numpy as np
class UAV(Actor):
    def __init__(self, params):
        super().__init__(params)
        # UAV specific properties
        self.max_azimuth_angle = params['max_azimuth_angle']
        self.coverage_radius = params['coverage_radius']
        self.group = params.get('group', [])
        self.group_mec = params.get('group_mec', [])
        self.max_safe_distance = params['max_safe_distance']
        self.profile_drag_coefficient = params['profile_drag_coefficient']
        self.solidity = params['solidity']
        self.rotor_dis_area = params['rotor_dis_area']
        self.blade_angular_velocity = params['blade_angular_velocity']
        self.rotor_radius = params['rotor_radius']
        self.incremental_correction = params['incremental_correction']
        self.UAV_weight = params['UAV_weight']
        self.blade_profile_power = params['blade_profile_power']
        self.induced_power = params['induced_power']
        self.tip_speed_rotor_blade = params['tip_speed_rotor_blade']
        self.mean_rotor_induced_power = params['mean_rotor_induced_power']
        self.velocity = params.get('velocity', [0, 0, 0])  # 默认值为[0, 0, 0]
        self.q_0 = params.get('q_0', [0, 0, 0])  # 默认值为[0, 0, 0]
        self.q_f = params.get('q_f', [0, 0, 0])  # 默认值为[0, 0, 0]
        self.trajectory = params.get('trajectory', [])  # 默认值为空列表
        self.max_bandwidth = params.get('max_bandwidth', 0)  # 默认值为0
        self.max_connection_num = params.get('max_connection_num', 0)  # 默认值为0
        self.task_list = params.get('task_list', [])  # 默认值为空列表
        self.request_vehicles = params.get('request_vehicles', [])  # 默认值为空列表
        self.preference_list = params.get('preference_list', [])  # 默认值为空列表
        self.queue_length = params.get('queue_length', 0)  # 默认值为0
        self.max_queue_length = params.get('max_queue_length', 0)  # 默认值为0
        self.unit_price = params['unit_price']
        self.max_unit_pricing = params['max_unit_pricing']

    def UpdateIniFinPosition(self, initial_pos, final_pos):
        self.q_0 = initial_pos
        self.q_f = final_pos

    def UpdateVelocity(self, velocity):
        self.velocity = velocity

    def UpdatePosition(self, position):
        self.position = position

    def UpdateTrajectory(self, cur_time, cur_velocity, cur_position):
        new_trajec = {'time': cur_time, 'vel': cur_velocity, 'pos': cur_position}
        self.trajectory.append(new_trajec)

    def ClearUAVCoverage(self):
        self.group = []
        self.group_mec = []

    def UpdateUAVCoverage(self, all_vehicles, all_mecs):
        print('-----------------------------')
        print(f'The vehicles within UAV {self.id} range:')
        coverage = []
        for vehicle in all_vehicles:
            r_distance = np.linalg.norm(np.array(self.position[:2]) - np.array(vehicle.position[:2]))
            if r_distance <= self.coverage_radius:
                coverage.append(vehicle.id)
                vehicle.set_belong_uav(self.id, r_distance)
                print(f' Vehicle {vehicle.id}, Distance: {r_distance}')
        self.group = coverage
        print('-----------------------------')
        print(f'The MECs within UAV {self.id} range:')
        coverage = []
        for mec in all_mecs:
            r_distance = np.linalg.norm(np.array(self.position[:2]) - np.array(mec.position[:2]))
            if r_distance <= self.coverage_radius:
                coverage.append(mec.id)
                mec.set_belong_uav(self.id, r_distance)
                print(f' MEC {mec.id}, Distance: {r_distance}')
        self.group_mec = coverage

    def SetQueue(self, queue):
        self.queue_length = queue

    def SetResource(self, cpu):
        self.CPU_freq = cpu
        self.avail_CPU = cpu

    def ClearRequestList(self):
        self.request_vehicles = []

    def SetUnitPrice(self, price):
        self.unit_price = price

    def ClearPreferenceList(self):
        self.preference_list = []

    def AddPreference(self, new_preference_struct):
        self.preference_list.append(new_preference_struct)

    def SortPreferenceList(self):
        self.preference_list.sort(key=lambda x: x['mutility'], reverse=True)

    def DeletePreference(self, delete_vid):
        self.preference_list = [item for item in self.preference_list if item['vid'] != delete_vid]

    def DeleteFromK(self, k):
        if k > len(self.preference_list):
            raise ValueError(f'k exceeds the preference list of UAV {self.id}. Cannot delete!')
        self.preference_list = self.preference_list[:k]
