'''
gym-copter Environment class with realistic physics

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym_copter.envs.copter_env import CopterEnv
from gym import spaces
import numpy as np

class CopterRealistic(CopterEnv):
    '''
    A class with continous state and action space.
    '''

    def __init__(self, dt=.001):

        CopterEnv.__init__(self)

        # Action space = motors
        self.action_space = spaces.Box(np.array([0,0,0,0]), np.array([1,1,1,1]))

        # Observation space = roll, pitch, yaw, altitude, groundspeed
        self.observation_space = spaces.Box(np.array([-90,-90,0,0,0]), np.array([+90,+90,359,1000,100])) 
