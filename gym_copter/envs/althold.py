'''
gym-copter Environment class for altitude-hold

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym_copter.envs.env import CopterEnv
from gym import spaces
import numpy as np

class CopterAltHold(CopterEnv):
    '''
    A GymCopter class with continous state and action space.
    Action space is [0,1], with all motors set to the same value.
    State space is altitude,vertical_velocity.
    Reward is proximity to a target altitude.
    '''

    # How close we must be to altitude target (m) to get reward
    TOLERANCE = 1.0

    def __init__(self, dt=.001, target=10, timeout=5):

        CopterEnv.__init__(self)

        self.dt = dt
        self.target = target
        self.timeout = timeout
        self.reward = 0

        # Action space = motors, rescaled from [0,1] to [-1,+1]
        self.action_space = spaces.Box(np.array([-1]), np.array([1]))

        # Observation space = altitude, vertical_velocity
        self.observation_space = spaces.Box(np.array([0,-np.inf]), np.array([np.inf,np.inf]))
        
    def step(self, action):

        # Rescale action from [-1,+1] to [0,1] and use it for all four motors
        motors = (1 + action[0] * np.ones(4)) / 2

        # Call parent-class step() to do basic update
        state, _, episode_over, info = CopterEnv.step(self, motors)

        # Dynamics uses NED coordinates, so negate to get altitude and vertical velocity
        altitude = -state[4]
        velocity = -state[5]

        # Accumulate reward for being close to altitude target, reset otherise
        self.reward = self.reward + 1 if abs(self.target-altitude) < self.TOLERANCE else 0

        # Too much time elapsed with no reward: set episode-over flag
        if self.reward == 0 and self.t > self.timeout:
            episode_over = True

        # Only one state; reward is altitude
        return (altitude,velocity), self.reward, episode_over, info

    def reset(self):
        CopterEnv.reset(self)
        return -self.state[4:6]
