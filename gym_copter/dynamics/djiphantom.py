#!/usr/bin/env python3
'''
Dynamics class for DJI Inspire

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym_copter.dynamics import Parameters
from gym_copter.dynamics.quadxap import QuadXAPDynamics


class DJIPhantomDynamics(QuadXAPDynamics):

    def __init__(self, framesPerSecond, g=QuadXAPDynamics.G):

        params = Parameters(

            # Estimated
            5.E-06,  # b force constatnt [F=b*w^2]
            2.E-06,  # d torque constant [T=d*w^2]

            # https:#www.dji.com/phantom-4/info
            1.380,  # mass [kg]
            0.350,  # arm length [m]

            # Estimated
            2,       # Ix [kg*m^2]
            2,       # Iy [kg*m^2]
            3,       # Iz [kg*m^2]
            38E-04,  # Jr prop inertial [kg*m^2]

            15000   # maxrpm
            )

        QuadXAPDynamics.__init__(self, params, framesPerSecond, g)
