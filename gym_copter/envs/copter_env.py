'''
gym-copter Environment classes

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym import Env, spaces
import numpy as np
from sys import stdout

from gym_copter.dynamics.phantom import DJIPhantomDynamics

class CopterEnv(Env):

    # Altitude threshold for being airborne
    ALTITUDE_MIN = 0.1

    metadata = {'render.modes': ['human']}

    def __init__(self, dt=.001):

        self.num_envs = 1
        self.dt = dt
        self.dynamics = DJIPhantomDynamics()
        self.hud = None
        self.state = np.zeros(12)
        self.ticks = 0

    def step(self, action):

        # Update dynamics and get kinematic state
        self.dynamics.setMotors(action)
        self.dynamics.update(self.dt)
        self.state = self.dynamics.getState()

        # Increment time count
        self.ticks += 1

        # Get return values 
        reward       = 0      # floating-point reward value from previous action
        episode_over = False  # whether it's time to reset the environment again (e.g., circle tipped over)
        info         = {}     # diagnostic info for debugging

        return self.state, reward, episode_over, info

    def reset(self):
        self.state = np.zeros(12)
        return self.state

    def render(self, mode='human'):

        from gym_copter.envs.hud import HUD

        if self.hud is None:

            self.hud = HUD()

        # Detect window close
        if not self.hud.isOpen(): return None

        return self.hud.display(mode,  self.state)

    def close(self):
        pass
