#!/usr/bin/env python3
'''
Dynamics class Mars Ingenuity copter

Copyright (C) 2021 Simon D. Levy, Alex Sender

MIT License
'''

from gym_copter.dynamics import MultirotorDynamics


# XXX This isn't really coaxial dynamics; its quadcopterX
class CoaxialDynamics(MultirotorDynamics):

    def __init__(self, vparams, framesPerSecond, wparams):

        MultirotorDynamics.__init__(self, vparams, 4, framesPerSecond, wparams)

    def u2(self,  o):
        '''
        roll right
        '''
        return (o[1] + o[2]) - (o[0] + o[3])

    def u3(self,  o):
        '''
        pitch forward
        '''
        return (o[1] + o[3]) - (o[0] + o[2])

    def u4(self,  o):
        '''
        yaw cw
        '''
        return (o[0] + o[1]) - (o[2] + o[3])

    def motorDirection(i):
        '''
        motor direction for animation
        '''
        dir = (-1, -1, +1, +1)
        return dir[i]


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

        # World parameters for Mars
        wparams = {'G': 3.721,    # Gravity
                   'rho': 0.017}   # Air density

        CoaxialDynamics.__init__(self, vparams, framesPerSecond, wparams)
