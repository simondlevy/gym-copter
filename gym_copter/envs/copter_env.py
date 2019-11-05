'''
Copyright (C) 2019 Simon D. Levy

MIT License
'''

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from gym_copter.dynamics.quadxap import QuadXAPDynamics
from gym_copter.dynamics import Parameters

class CopterEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, dt=.001):

        params = Parameters(

        # Estimated
        5.E-06, # b
        2.E-06, # d

        # https:#www.dji.com/phantom-4/info
        1.380,  # m (kg)
        0.350,  # l (meters)

        # Estimated
        2,      # Ix
        2,      # Iy
        3,      # Iz
        38E-04, # Jr
        15000)  # maxrpm

        self.action_space = spaces.Box( np.array([0,0,0,0]), np.array([1,1,1,1]))  # motors

        self.dt = dt

        self.copter = QuadXAPDynamics(params)

    def step(self, action):

        obj          = None # an environment-specific object representing your observation of the environment
        reward       = 0.0 # floating-point reward value from previous action
        episode_over = False # whether it's time to reset the environment again (e.g., pole tipped over)
        info         = {}    # diagnostic info for debugging

        self.copter.setMotors(action)

        self.copter.update(self.dt)

        return obj, reward, episode_over, info

    def reset(self):
        pass

    def render(self, mode='human'):

        print(self.copter.getState().pose.location)

    def close(self):
        pass
