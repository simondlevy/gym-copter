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

    def __init__(self, dt=.001, target=10, timeout=10, tolerance=1.0, maxvel=1.0):

        CopterEnv.__init__(self)

        self.dt = dt
        self.target = target
        self.timeout = timeout
        self.tolerance = tolerance
        self.maxvel = maxvel

        # Action space = motors, rescaled from [0,1] to [-1,+1]
        self.action_space = spaces.Box(np.array([-1]), np.array([1]))

        # Observation space = altitude, vertical_velocity
        self.observation_space = spaces.Box(np.array([0,-np.inf]), np.array([np.inf,np.inf]))
        
    def step(self, action):

        # Rescale action from [-1,+1] to [0,1] and use it for all four motors
        motors = (1 + action[0] * np.ones(4)) / 2

        # Call parent-class step() to do basic update
        state = CopterEnv._get_state(self, motors)

        # Defaults for return
        done = False
        reward = 0
        info = {}

        # Dynamics uses NED coordinates, so negate to get altitude and vertical velocity
        altitude = -state[4]
        velocity = -state[5]

        # Reward for being close to altitude target at low velocity
        if abs(self.target-altitude) < self.tolerance and abs(velocity) < self.maxvel:
            reward = 1
            done = True

        # Too much time elapsed: set episode-over flag
        if self.t > self.timeout:
            done = True

        # Only one state; reward is altitude
        return (altitude,velocity), reward, done, info

    def reset(self):
        print('reset')
        CopterEnv.reset(self)
        return -self.state[4:6]
