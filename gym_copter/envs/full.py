'''
gym-copter Environment class for full state-space kinematics

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym_copter.envs.env import CopterEnv
from gym import spaces
import numpy as np

class CopterFull(CopterEnv):
    '''
    A GymCopter class with continous state and action space.
    Action space is [0,1]*4.
    State space is full 12-dimensional kinematic state vector from dynamics.
    Reward is zero (stubbed).
    '''

    def __init__(self, dt=.001):

        CopterEnv.__init__(self)

        self.dt = dt

        # Action space = motors
        self.action_space = spaces.Box(np.zeros(4), np.ones(4))

        # Observation space (XXX should limit angles to -pi/+pi)
        self.observation_space = spaces.Box(-np.inf*np.ones(12), +np.inf*np.ones(12))
        
    def step(self, action):

        # Call parent-class step() to do basic update
        return CopterEnv.step(self, action)

    def reset(self):
        CopterEnv.reset(self)
        return 0

