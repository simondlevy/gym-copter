'''
gym-copter Environment class with simplified physics

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym_copter.envs.copter_env import CopterEnv
from gym import spaces
import numpy as np

class CopterSimple(CopterEnv):
    '''
    A simplified copter class for Q-Learning.
    Observation space: on-ground (0) or airborne (1)
    Action space:      motors=0 or motors=1
    Reward             altitude
    '''

    TIMEOUT      = 10.0

    def __init__(self):

        CopterEnv.__init__(self)

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Discrete(2)

    def step(self, action):

        # Convert discrete action index to array of floating-point number values
        motors = [float(action)]*4

        # Call parent-class step() to do basic update
        state, reward, episode_over, info = CopterEnv.step(self, motors)

        # Dynamics uses NED coordinates, so negate to get altitude
        altitude = -state[4]

        # Too many ticks elapsed: set episode-over flag
        if self.ticks*self.dt > self.TIMEOUT:
            episode_over = True

        return int(altitude>.1), altitude, episode_over, info

    def reset(self):
        CopterEnv.reset(self)
        return np.array([0])

