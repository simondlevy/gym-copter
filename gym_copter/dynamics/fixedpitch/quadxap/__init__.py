'''
Dynamics class for quad-X frames using ArduPilot motor layout:

    3cw   1ccw

        ^

    2ccw  4cw

Copyright (C) 2019 Simon D. Levy

MIT License
'''


from gym_copter.dynamics.fixedpitch import FixedPitchDynamics


class QuadXAPDynamics(FixedPitchDynamics):

    def __init__(self, params, framesPerSecond):

        FixedPitchDynamics.__init__(self, params, framesPerSecond)

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
