'''
    Vehicle constants for DJI Phantom  

    Copyright (C) 2023 Simon D. Levy

    MIT License
'''

vehicle_params = {

    # Estimated
    'B': 5.E-03,  # force constatnt [F=b*w^2]
    'D': 2.E-06,  # torque constant [T=d*w^2]

    # https:#www.dji.com/phantom-4/info
    'M': 1.380,  # mass [kg]
    'L': 0.350,  # arm length [m]

    # Estimated
    'Ix': 2,       # [kg*m^2]
    'Iy': 2,       # [kg*m^2]
    'Iz': 3,       # [kg*m^2]
    'Jr': 38E-04,  # prop inertial [kg*m^2]

    'maxrpm': 15000
}
