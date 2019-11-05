'''
Dynamics class for quad-X frames using ArduPilot motor layout:

    3cw   1ccw
       \ /
        ^
       / \
    2ccw  4cw

Copyright (C) 2019 Simon D. Levy

MIT License
'''


#include "MultirotorDynamics.hpp"

from gym_copter.dynamics import MultirotorDynamics

class QuadXAPDynamics(MultirotorDynamics):

    def __init__(self, params):

        MultirotorDynamics.__init__(self, params, 4)

    def u2(self,  o):
        '''
        roll right
        '''
        return (o[1] + o[2]) - (o[0] + o[3]);
   

    def u3(self,  o):
        '''
        pitch forward
        '''
        return (o[1] + o[3]) - (o[0] + o[2]);
   

    def u4(self,  o):
        '''
        yaw cw
        '''
        return (o[0] + o[1]) - (o[2] + o[3]);

    def motorDirection(i):
        '''
        motor direction for animation
        '''
        dir = (-1, -1, +1, +1)
        return dir[i];
