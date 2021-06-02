#!/usr/bin/env python3
'''
Dynamics class Mars Ingenuity copter

Copyright (C) 2021 Simon D. Levy, Alex Sender

MIT License
'''

from gym_copter.dynamics.coaxial import CoaxialDynamics


class IngenuityDynamics(CoaxialDynamics):

    def __init__(self, framesPerSecond):

        # Vehicle parameters for Ingenuity [See Bouabdallah et al. (2004)]
        vparams = {

            # Estimated
            'B': 5.E-06,  # force constatnt [F=b*w^2]
            'D': 2.E-06,  # torque constant [T=d*w^2]

            # https:#www.dji.com/phantom-4/info
            'M': 1.380,  # mass [kg]
            'L': 0.350,  # arm length [m]
            'C_L': 0.4,  # lift coeifficent: need to add this as a function
                         # (found graph that I need to quantify)

            # Estimated
            'Ix': 2,       # [kg*m^2]
            'Iy': 2,       # [kg*m^2]
            'Iz': 3,       # [kg*m^2]
            'Jr': 38E-04,  # prop inertial [kg*m^2]

            'maxrpm': 15000
            }

        CoaxialDynamics.__init__(self, vparams, framesPerSecond)

        # World parameters for Mars
        self.setWorldParams(3.721, 0.017)
