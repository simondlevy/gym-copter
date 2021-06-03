'''
Dynamics classes for UAVs with coaxial rotors (e.g. Ingenuity)

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from gym_copter.dynamics import MultirotorDynamics


class CoaxialDynamics(MultirotorDynamics):
    '''
    Order is: rotor1, rotor2, servo1, servo2.
    '''

    def __init__(self, vparams, framesPerSecond, wparams):

        MultirotorDynamics.__init__(self, vparams, framesPerSecond, wparams)

    def _u2(self,  o):
        '''
        roll right
        '''
        return o[2]

    def _u3(self,  o):
        '''
        pitch forward
        '''
        return o[3]

    def _u4(self,  o):
        '''
        yaw cw
        '''
        return o[0] - o[1]

    def _getThrusts(self, u1, u2, u3):

        return u1, u2, u3  # XXX
