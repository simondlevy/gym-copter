'''
Copyright (C) 2019 Simon D. Levy

MIT License
'''

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from gym_copter.dynamics.quadxap import QuadXAPDynamics

class CopterEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.action_space = spaces.Box( np.array([0,0,0,0]), np.array([1,1,1,1]))  # motors

    def step(self, action):

        obj          = None # an environment-specific object representing your observation of the environment
        reward       = 0.0 # floating-point reward value from previous action
        episode_over = False # whether it's time to reset the environment again (e.g., pole tipped over)
        info         = {}    # diagnostic info for debugging

        return obj, reward, episode_over, info

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass
