#!/usr/bin/env python3
'''
Dynamics class for Ingenuity Mars copter

XXX for now we just mock it up with DJI Phantom dynamics

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from gym_copter.dynamics.quadxap import QuadXAPDynamics


class IngenuityDynamics(QuadXAPDynamics):

    def __init__(self, framesPerSecond):

        # See Bouabdallah et al. (2004)
        vparams = {

            # Estimated
            'B': 5.E-06,  # force constatnt [F=b*w^2]
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

        QuadXAPDynamics.__init__(self, vparams, framesPerSecond)
