'''
gym-copter Environment superclass

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym import Env, spaces
import numpy as np
from time import time

from gym_copter.dynamics.djiphantom import DJIPhantomDynamics

class CopterEnv(Env):

    # Default time constant
    DELTA_T = 0.001

    def __init__(self):

        self.num_envs = 1
        self.hud = None

        # We handle time differently if we're rendering
        self.dt = CopterEnv.DELTA_T

        self._init()

    def step(self, action):

        # Update dynamics and get kinematic state
        self.dynamics.setMotors(action)
        self.dynamics.update(self.dt)
        self.state = self.dynamics.getState()

        # Get return values 
        reward       = 0      # floating-point reward value from previous action
        episode_over = False  # whether it's time to reset the environment again
        info         = {}     # diagnostic info for debugging

        # Accumulate time
        self.t += self.dt

        return self.state, reward, episode_over, info

    def reset(self):
        self._init()
        return self.state

    def render(self, mode='hud'):

        from gym_copter.envs.hud import HUD

        if self.hud is None:
            self.hud = HUD()

        # Track time
        tcurr = time()
        self.dt = (tcurr - self.tprev) if self.tprev > 0 else CopterEnv.DELTA_T
        self.tprev = tcurr
 
        # Detect window close
        if not self.hud.isOpen(): return None

        return self.hud.display(mode, self.state)

    def close(self):
        Env.close(self)        

    def time(self):

        return self.t

    def _init(self):
        
        self.dynamics = DJIPhantomDynamics()
        self.state = np.zeros(12)
        self.tprev = 0
        self.t = 0
