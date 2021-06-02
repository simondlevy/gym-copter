'''
Dynamics classes for UAVs with coaxial rotors (e.g. Ingenuity)

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from gym_copter.dynamics import MultirotorDynamics


class CoaxialDynamics(MultirotorDynamics):
    '''
    XXX This isn't really coaxial dynamics; its quadcopterX
    '''

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
