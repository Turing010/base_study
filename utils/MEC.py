import numpy as np
from utils.Actor import Actor

class MEC(Actor):
    def __init__(self, params):
        super().__init__(params)
        self.coverage_radius = params['coverage_radius']
        self.group = params['group']
        self.belong_to_UAV = []
        self.max_bandwidth = params['max_bandwidth']
        self.max_connection_num = params['max_connection_num']
        self.task_list = []
        self.request_vehicles = []
        self.preference_list = []
        self.queue_length = params['queue_length']
        self.max_queue_length = params['max_queue_length']
        self.unit_price = params['unit_price']
        self.max_unit_pricing = params['max_unit_pricing']

    def setBelongUAV(self, belong_uav_id, distance):
        new_belong_to_UAV = {'belong_server_id': belong_uav_id, 'distance': distance}
        self.belong_to_UAV.append(new_belong_to_UAV)

    def ClearBelongUAV(self):
        self.belong_to_UAV = []

    def sortBelongUAV(self):
        self.belong_to_UAV.sort(key=lambda x: x['distance'])

    def ClearMECCoverage(self):
        self.group = []

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
        self.preference_list = [x for x in self.preference_list if x['vid'] != delete_vid]

    def DeleteFromK(self, k):
        if k > len(self.preference_list):
            raise Exception("k exceeds the preference list of MEC {}. Cannot delete!".format(self.id))
        self.preference_list = self.preference_list[:k]

    def UpdateMECCoverage(self, all_Vehicles):
        coverage = []
        print("MEC id = {}, position = ({}, {})".format(self.id, self.position[0], self.position[1]))
        print("The vehicles within MEC {} range:".format(self.id))

        for vehicle in all_Vehicles:
            r_distance = np.linalg.norm([self.position[0], self.position[1]] - [vehicle.position[0], vehicle.position[1]])
            if r_distance <= self.coverage_radius:
                coverage.append(vehicle.id)
                vehicle.setBelongMEC(self.id, r_distance) # save the MEC id
                print("Vehicle {}, Horizontal Distance: {}".format(vehicle.id, r_distance))

        if not self.group:
            print("There is no vehicle in the range of MEC {}".format(self.id))
