'''
Dynamics classes for UAVs with coaxial rotors (e.g. Ingenuity)

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from gym_copter.dynamics import MultirotorDynamics


class CoaxialDynamics(MultirotorDynamics):

    def __init__(self, vparams, framesPerSecond):

        MultirotorDynamics.__init__(self, vparams, 2, framesPerSecond)

    def _u2(self,  o):
        '''
        roll right
        '''
        return 0  # XXX

    def _u3(self,  o):
        '''
        pitch forward
        '''
        return 0  # XXX

    def _u4(self,  o):
        '''
        yaw cw
        '''
        return (o[0] - o[1])

    def _getThrusts(self, u1, u2, u3):

        # return self.B * u1, self.L * self.B * u2, self.L * self.B * u3
        return u1, u2, u3  # XXX
