'''
gym-copter Environment class for takeoff

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym_copter.envs.env import CopterEnv
from gym import spaces
import numpy as np

class CopterAltHold(CopterEnv):
    '''
    A GymCopter class with continous state and action space.
    Action space is [-1,+1], translated to [0,1], with all motors set to the
    same value.  State space is altitude,vertical_velocity.
    Copter starts at target altitude.  Reward is how long it remains there.
    '''

    def __init__(self, dt=.001, target=10, tolerance=1.0):

        CopterEnv.__init__(self, dt)

        self.target = target
        self.tolerance = tolerance

        # Action space = motors, rescaled from [0,1] to [-1,+1]
        self.action_space = spaces.Box(np.array([-1]), np.array([1]))

        # Observation space = altitude, vertical_velocity
        self.observation_space = spaces.Box(np.array([0,-np.inf]), np.array([np.inf,np.inf]))

        self._init()

    def step(self, action):

        # Rescale action from [-1,+1] to [0,1]
        motor = (1 + action[0]) / 2

        # Use it for all four motors
        motors = motor * np.ones(4)

        # Call parent-class step() to do basic update
        CopterEnv._update(self, motors)

        # Dynamics uses NED coordinates, so negate to get altitude and vertical velocity
        altitude = -self.state[4]
        velocity = -self.state[5]

        # Get a reward for every timestep on target
        self.reward += (int)(abs(altitude-self.target) < self.tolerance)

        # False = max_episodes in registry determines whether we're done
        return (altitude,velocity), self.reward, False, {}

    def reset(self):
        CopterEnv.reset(self)
        self._init()
        return -self.state[4:6]

    def _init(self):
        self.dynamics.setState((0,0,0,0,-float(self.target),0,0,0,0,0,0,0))
        self.reward = 0
