import numpy as np
class Actor:
    def __init__(self, params):
        # Basic
        self.id = params['id']
        self.type = params['type']
        self.weight_factor = params['weight_factor']
        self.discount_factor_bargain = params['discount_factor_bargain']

        # Mobility
        self.position = np.array(params['position'])##三维，源代码为pos=[0,0,0]

        # Communication params
        self.trans_power_dbm = params['trans_power_dbm']
        self.trans_power_w = 10 ** (params['trans_power_dbm'] / 10) / 1000  # unit conversion dBm-w
        self.receive_sensitivity_dbm = params['receive_sensitivity_dbm']
        self.receive_sensitivity_w = 10 ** (params['receive_sensitivity_dbm'] / 10) / 1000  # unit conversion dBm-w

        # Computation params
        self.CPU_freq = params['CPU_freq']
        self.CPU_chip_params = params['CPU_chip_params']
        self.avail_CPU = params['CPU_freq']
        self.energy_constraint = params['energy_constraint']

    def set_cpu(self, cpu):
        self.CPU_freq = cpu

    def set_trans_power(self, trans_power):
        self.trans_power_dbm = trans_power
        self.trans_power_w = 10 ** (trans_power / 10) / 1000