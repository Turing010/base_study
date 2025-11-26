import numpy as np
from utils.MEC import MEC
from utils.UAV import UAV
from utils.Vehicle import Vehicle

def HelperInitialize(base):

    # MEC parameters initialization
    def InitializeMecParams():
        m_params = {}
        # Common
        m_params['type'] = 'MEC'
        m_params['weight_factor'] = base.mec_weight
        m_params['discount_factor_bargain'] = base.discount_factor_bargain
        # mobility
        m_params['position'] = np.array([0, 0, base.MEC_hight])
        # Communication params
        m_params['trans_power_dbm'] = 30
        m_params['receive_sensitivity_dbm'] = -84
        # computation
        m_params['CPU_freq'] = base.mec_cpu_freq
        m_params['CPU_chip_params'] = base.CPU_chip_params
        m_params['energy_constraint'] = 3.6e-7 * m_params['CPU_freq']
        # MEC class
        m_params['coverage_radius'] = base.MEC_coverage_range
        m_params['group'] = []
        m_params['belong_to_UAV'] = {'belong_server_id': [], 'distance': []}
        # communication
        m_params['max_bandwidth'] = base.mec_bandwidth
        m_params['max_connection_num'] = 20
        m_params['queue_length'] = base.mec_queue_length
        m_params['max_queue_length'] = base.mec_queue_length
        m_params['unit_price'] = base.MEC_unit_price
        m_params['max_unit_pricing'] = base.max_unit_pricing
        return m_params

    # Initialize MEC server parameters
    MEC_params = [InitializeMecParams() for _ in range(base.all_MECs_num)]

    # UAV parameters initialization
    def InitializeUavParams():
        u_params = {}
        u_params['type'] = 'UAV'
        u_params['weight_factor'] = base.uav_weight
        u_params['discount_factor_bargain'] = base.discount_factor_bargain
        # mobility
        u_params['position'] = np.array([0, 0, base.UAV_altitude])
        u_params['velocity'] = 0
        u_params['q_0'] = np.array([0, 0, base.UAV_altitude])
        u_params['q_f'] = np.array([0, 0, base.UAV_altitude])
        # communication
        u_params['trans_power_dbm'] = 30
        u_params['receive_sensitivity_dbm'] = -84
        # computation
        u_params['CPU_freq'] = base.uav_cpu_freq
        u_params['CPU_chip_params'] = base.CPU_chip_params
        u_params['energy_constraint'] = 3 * 3.6e6
        # UAV class
        u_params['group'] = []
        u_params['group_mec'] = []
        u_params['max_azimuth_angle'] = np.pi / 3
        u_params['coverage_radius'] = base.UAV_coverage_range
        # communication
        u_params['max_bandwidth'] = base.uav_bandwidth
        u_params['max_connection_num'] = 10
        # computation
        u_params['unit_price'] = base.UAV_unit_price
        u_params['max_unit_pricing'] = base.max_unit_pricing
        u_params['queue_length'] = base.uav_queue_length
        u_params['max_queue_length'] = base.uav_queue_length
        # UAV feature -- constant
        u_params['max_safe_distance'] = 10
        u_params['profile_drag_coefficient'] = base.profile_drag_coefficient
        u_params['solidity'] = base.solidity
        u_params['rotor_dis_area'] = base.rotor_dis_area
        u_params['blade_angular_velocity'] = base.blade_angular_velocity
        u_params['rotor_radius'] = base.rotor_radius
        u_params['incremental_correction'] = base.incremental_correction
        u_params['tip_speed_rotor_blade'] = base.tip_speed_rotor_blade
        u_params['UAV_weight'] = base.UAV_weight
        u_params['mean_rotor_induced_power'] = base.mean_rotor_induced_power
        # Blade profile power
        u_params['blade_profile_power'] = (u_params['profile_drag_coefficient'] / 8) * base.air_density * \
            u_params['solidity'] * u_params['rotor_dis_area'] * \
            (u_params['blade_angular_velocity'] ** 3) * \
            (u_params['rotor_radius'] ** 3)
        # Induced power
        u_params['induced_power'] = (1 + u_params['incremental_correction']) * \
            (u_params['UAV_weight'] ** 1.5) / \
            np.sqrt(2 * base.air_density * u_params['rotor_dis_area'])
        return u_params

    # Initialize UAV parameters
    UAV_params = [InitializeUavParams() for _ in range(base.all_UAVs_num)]

    # Vehicle parameters initialization
    def InitializeVehicleParams():
        v_params = {}
        # character
        v_params['type'] = 'Vehicle'
        v_params['weight_factor'] = base.veh_weight
        v_params['discount_factor_bargain'] = base.discount_factor_bargain
        # mobility
        v_params['position'] = np.array([0, 0, 1])
        v_params['velocity'] = base.veh_mean_velocity
        v_params['direction'] = base.veh_mean_direction
        # communication
        v_params['trans_power_dbm'] = 10 + 15 * np.random.rand()
        v_params['receive_sensitivity_dbm'] = -84
        # computation
        v_params['CPU_freq'] = 0.5e9 + 0.1e9 * np.random.rand()
        v_params['CPU_chip_params'] = base.CPU_chip_params
        v_params['belong_to_MEC'] = {'belong_server_id': [], 'distance': []}
        v_params['belong_to_UAV'] = {'belong_server_id': [], 'distance': []}
        v_params['task_arrival_rate'] = 0.6 + 0.4 * np.random.rand()
        v_params['count_delay'] = 0
        v_params['best_off_probability'] = [1, 1]
        v_params['max_budget'] = 20
        v_params['energy_constraint'] = 3.6e-7 * v_params['CPU_freq']
        v_params['queue_length'] = 1
        return v_params

    # Initialize vehicle parameters
    Vehicle_params = [InitializeVehicleParams() for _ in range(base.all_Vehicles_num)]

    all_MECs, all_Vehicles, all_UAVs = [], [], []

    # Initialize MEC server
    for i, params in enumerate(MEC_params):
        params['id'] = i
    all_MECs = [MEC(params) for params in MEC_params]

    # Initialize vehicle
    for i, params in enumerate(Vehicle_params):
        params['id'] = i
    all_Vehicles = [Vehicle(params, base) for params in Vehicle_params]

    # Initialize UAV
    for i, params in enumerate(UAV_params):
        params['id'] = i
    all_UAVs = [UAV(params) for params in UAV_params]

    return all_Vehicles, all_MECs, all_UAVs

# Set different CPU frequencies for UAVs
# all_UAVs[0].CPU_freq = 10 * 1e9
# all_UAVs[1].CPU_freq = 30 * 1e9
# all_UAVs[2].CPU_freq = 40 * 1e9
# all_UAVs[3].CPU_freq = 50 * 1e9
# all_UAVs[4].CPU_freq = 10 * 1e9
# all_UAVs[5].CPU_freq = 30 * 1e9
# all_UAVs[6].CPU_freq = 30 * 1e9

# all_UAVs[3].queue_length = 20
# all_UAVs[7].CPU_freq = 50 * 1e9

