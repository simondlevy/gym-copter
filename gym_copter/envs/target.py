'''
gym-copter Environment class for visually-guided targeting

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from gym_copter.envs.env import CopterEnv
from gym import spaces
import numpy as np

class CopterTarget(CopterEnv):
    '''
    A GymCopter class with continous state and action space.
    Action space is [-1,+1]^4, translated to [0,1]^4
    State space is full 12-dimensional state space from dynamics, plus target position.
    Reward is hitting prey.
    '''

    def __init__(self):

        CopterEnv.__init__(self, statedims=15)

        # Initialize state
        self._init()

        # Action space = motors, rescaled from [0,1] to [-1,+1]
        self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))

        # Observation space = full state space plus target position
        self.observation_space = spaces.Box(np.array([-np.inf]*15), np.array([+np.inf]*15))

        self.target_theta = 0

    def step(self, action):

        # Rescale action from [-1,+1] to [0,1]
        motors = (1 + action) / 2

        # Call parent-class method to do basic state update, return whether vehicle crashed
        crashed = CopterEnv._update(self, motors)

        # Update target position
        self.state[12] = 10 * np.cos(self.target_theta)
        self.state[13] = 10 * np.sin(self.target_theta)
        self.target_theta += .01

        print(self.state[12:])

        # Fake up reward for now
        reward = 0

        return self.state, reward, crashed, {}

    def reset(self):
        CopterEnv.reset(self)
        self._init()
        return self.state

    def _init(self):
        self.state[14] = 10 # target altitude (m)


