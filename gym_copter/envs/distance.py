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

    def __init__(self, timeout=10):

        CopterEnv.__init__(self)

        self.timeout = timeout

        # Initialize state
        self._init()

        # Action space = motors, rescaled from [0,1] to [-1,+1]
        self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))

        # Observation space = altitude, vertical_velocity
        self.observation_space = spaces.Box(np.array([-np.inf]*12), np.array([+np.inf]*12))

        self.count = 0

    def step(self, action):

        # Rescale action from [-1,+1] to [0,1]
        motors = (1 + action) / 2

        self.count += 1

        # Call parent-class method to do basic state update
        CopterEnv._update(self, motors)

       # Integrate position
        self.position += self.state[0:5:2]

        # Reward is logarithm of Euclidean distance from origin
        reward = np.sqrt(np.sum(self.position[0:2]**2))

        # Quit if we haven't moved after a specified amount of time
        done = self.t > self.timeout #and distance == 0

        return self.state, reward, done, {}

    def reset(self):
        CopterEnv.reset(self)
        self.position = np.zeros(3)
        self.count = 0
        return self.state

