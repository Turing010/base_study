import random
import numpy as np
def InitializeMap(base, all_Vehicles, all_MECs, all_UAVs):

    # Set initial velocity of all UAVs to 0
    for uav in all_UAVs:
        uav.velocity = 0

    # Update initial and final positions of UAVs
    all_UAVs[0].UpdateIniFinPosition(np.array([300, 700, base.UAV_altitude]), np.array([1000, 0, base.UAV_altitude]))
    all_UAVs[1].UpdateIniFinPosition(np.array([700, 700, base.UAV_altitude]), np.array([0, 0, base.UAV_altitude]))
    all_UAVs[2].UpdateIniFinPosition(np.array([300, 300, base.UAV_altitude]), np.array([1000, 1000, base.UAV_altitude]))
    all_UAVs[3].UpdateIniFinPosition(np.array([700, 300, base.UAV_altitude]), np.array([0, 1000, base.UAV_altitude]))

    # Update initial positions of UAVs
    all_UAVs[0].UpdatePosition(np.array([300, 700, 100]))
    all_UAVs[1].UpdatePosition(np.array([700, 700, 100]))
    all_UAVs[2].UpdatePosition(np.array([300, 300, 100]))
    all_UAVs[3].UpdatePosition(np.array([700, 300, 100]))

    # Update position of the first MEC
    all_MECs[0].position = np.array([500, 500, 0])

    # Set velocity of all UAVs to 0 again
    for uav in all_UAVs:
        uav.velocity = 0

    # Update positions of all vehicles with random values
    for vehicle in all_Vehicles:
        vehicle.UpdatePosition(np.array([300 + 350 * random.random(), 300 + 370 * random.random(), 0]))

    # Uncomment the following lines to replicate the rest of the MATLAB code in Python
    # for i in range(10):
    #     all_Vehicles[i].UpdatePosition([280 + 40 * random.random(), 680 + 30 * random.random(), 0])
