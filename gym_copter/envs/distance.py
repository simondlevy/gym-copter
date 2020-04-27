'''
gym-copter Environment class for maximizing distance

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from gym_copter.envs.env import CopterEnv
from gym import spaces
import numpy as np

class CopterDistance(CopterEnv):
    '''
    A GymCopter class with continous state and action space.
    Action space is [-1,+1]^4, translated to [0,1]^4
    State space is full 12-dimensional state space from dynamics.
    Reward is distance traveled.
    '''

    def __init__(self):

        CopterEnv.__init__(self)

        # Initialize state
        self._init()

        # Action space = motors, rescaled from [0,1] to [-1,+1]
        self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))

        # Observation space = full state space
        self.observation_space = spaces.Box(np.array([-np.inf]*12), np.array([+np.inf]*12))

    def step(self, action):

        # Rescale action from [-1,+1] to [0,1]
        motors = (1 + action) / 2

        # Call parent-class method to do basic state update, return whether vehicle crashed
        crashed = CopterEnv._update(self, motors)

        # Integrate position
        self.position += self.state[0:5:2]

        # Reward is logarithm of Euclidean distance from origin
        reward = np.sqrt(np.sum(self.position[0:2]**2))

        return self.state, reward, crashed, {}

    def reset(self):
        CopterEnv.reset(self)
        self._init()
        return self.state

    def _init(self):
        self.position = np.zeros(3)


