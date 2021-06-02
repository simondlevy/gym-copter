#!/usr/bin/env python3
'''
Dynamics class Mars Ingenuity copter

Copyright (C) 2021 Simon D. Levy, Alex Sender

MIT License
'''

from gym_copter.dynamics import MultirotorDynamics


# XXX This isn't really coaxial dynamics; its quadcopterX
class CoaxialDynamics(MultirotorDynamics):

    def __init__(self, vparams, framesPerSecond):

        MultirotorDynamics.__init__(self, vparams, 4, framesPerSecond)

    def _u2(self,  o):
        '''
        roll right
        '''
        return (o[1] + o[2]) - (o[0] + o[3])

    def _u3(self,  o):
        '''
        pitch forward
        '''
        return (o[1] + o[3]) - (o[0] + o[2])

    def _u4(self,  o):
        '''
        yaw cw
        '''
        return (o[0] + o[1]) - (o[2] + o[3])

    def _getThrusts(self, u1, u2, u3):

        return self.B * u1, self.L * self.B * u2, self.L * self.B * u3

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


