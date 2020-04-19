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

        # Rescale action from [-1,+1] to [0,1]
        motor = (1 + action[0]) / 2

        # Use it for all four motors
        motors = motor * np.ones(4)

        # Call parent-class step() to do basic update
        state = CopterEnv._update(self, motors)

        # Dynamics uses NED coordinates, so negate to get altitude and vertical velocity
        altitude = -state[4]
        velocity = -state[5]

        # Reward for being close to altitude target at low velocity
        #if abs(self.target-altitude) < self.tolerance: #and abs(velocity) < self.maxvel:
        #    reward = 1
        #    done = True

        costs = (altitude-self.target)**2 + .1*velocity**2 + .001*(motor**2)

        # Too much time elapsed: set episode-over flag
        if self.t > self.timeout:
            done = True

        # Only one state; reward is altitude
        return (altitude,velocity), -costs, False, {}

    def reset(self):
        CopterEnv.reset(self)
        return -self.state[4:6]
