import numpy as np
from utils.Actor import Actor
from utils.Base import Base
from utils.Task import Task
class Vehicle(Actor):
    def __init__(self, params, base):
        super().__init__(params)
        self.base = base
        self.belong_to_MEC = []
        self.belong_to_UAV = []
        self.strategy = 0
        self.preference_list = []
        self.task_arrival_rate = params['task_arrival_rate']
        self.best_off_probability = params['best_off_probability']
        self.count_delay = 0
        self.max_budget = params['max_budget']
        self.velocity = params['velocity']
        self.direction = params['direction']
        self.trajectory = []
        self.up_data_rate_MEC = []
        self.up_data_rate_UAV = []
        self.queue_length = params['queue_length']

    def setBelongMEC(self, belong_mec_id, distance):
        new_belong_to_MEC = {'belong_server_id': belong_mec_id, 'distance': distance}
        self.belong_to_MEC.append(new_belong_to_MEC)

    def sortBelongMEC(self):
        self.belong_to_MEC.sort(key=lambda x: x['distance'])

    def setBelongUAV(self, belong_uav_id, distance):
        new_belong_to_UAV = {'belong_server_id': belong_uav_id, 'distance': distance}
        self.belong_to_UAV.append(new_belong_to_UAV)

    def sortBelongUAV(self):
        self.belong_to_UAV.sort(key=lambda x: x['distance'])

    def ClearBelongMECServer(self):
        self.belong_to_MEC = []

    def ClearBelongUAVServer(self):
        self.belong_to_UAV = []

    def setUpDataRateMEC(self, up_data_rate_struct):
        self.up_data_rate_MEC.append(up_data_rate_struct)

    def setUpDataRateUAV(self, up_data_rate_struct):
        self.up_data_rate_UAV.append(up_data_rate_struct)

    def clearUpDataRate(self):
        self.up_data_rate_MEC = []
        self.up_data_rate_UAV = []

    def LocalTaskProccessingUpdate(self, run_slot_time, sum_strategy):
        if (self.strategy == 1 or self.strategy == sum_strategy + 1) and self.task.state == 1:
            if hasattr(self, 'local_remain_delay') and self.local_remain_delay > 0:
                self.local_remain_delay -= run_slot_time
                print(f'Vehicle {self.id}: The remaining processing time for Local execution is : {self.local_remain_delay}')
            else:
                self.task.state = 2
                print(f'Vehicle {self.id} Local execution has been fulfilled')

    def SetLocalRemainDelay(self, delay):
        self.local_remain_delay = delay

    def setTask(self, task_size, cpu_per_bit):
        self.task.size = task_size
        self.task.cpu_per_bit = cpu_per_bit
        self.task.cpu = task_size * cpu_per_bit

    def UpdteCurrOffPro(self, probability):
        self.best_off_probability[1] = probability

    def SetOldOffPro(self, probability):
        self.best_off_probability[0] = probability

    def UpdteMaxDelay(self, time_slot):
        self.task.max_delay -= time_slot

    def SetFinishCountDelay(self):
        self.count_delay = 1

    def SetUnitPay(self, pay):
        self.task.unit_pay = pay

    def ClearPreferenceList(self):
        self.preference_list = []

    def SetStrategy(self, strategy):
        self.strategy = strategy

    def CPUAllocate(self, allocate):
        self.task.allocate = allocate

    def UpdateVelocity(self, velocity):
        self.velocity = velocity

    def UpdateDirection(self, direction):
        self.direction = direction

    def UpdatePosition(self, position):
        self.position = position

    def UpdateTrajectory(self, cur_time, cur_position, cur_velocity, cur_direction):
        new_trajec = {'time': cur_time, 'pos': cur_position, 'vel': cur_velocity, 'dir': cur_direction}
        self.trajectory.append(new_trajec)

    def AddPreference(self, new_preference_struct):
        self.preference_list.append(new_preference_struct)

    def SortPreferenceList(self):
        self.preference_list.sort(key=lambda x: x['vutility'], reverse=True)

    def DeletePreference(self, delete_mid):
        self.preference_list = [x for x in self.preference_list if x['mid'] != delete_mid]
