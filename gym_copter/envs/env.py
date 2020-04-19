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

    def __init__(self, dt=0.001):

        self.num_envs = 1
        self.hud = None

        # We handle time differently if we're rendering
        self.dt = dt

        self._init()

    def _get_state(self, action):

        # Update dynamics and get kinematic state
        self.dynamics.setMotors(action)
        self.dynamics.update(self.dt)
        self.state = self.dynamics.getState()

        # Get return values 
        reward       = 0      # floating-point reward value from previous action
        done = False  # whether it's time to reset the environment again
        info         = {}     # diagnostic info for debugging

        # Accumulate time
        self.t += self.dt

        return self.state

    def reset(self):

        self._init()
        return self.state

    def render(self, mode='hud'):

        # Track time
        tcurr = time()
        self.dt = (tcurr - self.tprev) if self.tprev > 0 else self.dt
        self.tprev = tcurr

        # Support various modes
        if mode == 'hud':
            return self._render_hud(mode)
        elif mode.lower() == '3d':
            return self._render_3d(mode)
        else:
            raise Exception('Unsupported render mode ' + mode)

    def close(self):

        Env.close(self)        

    def time(self):

        return self.t

    def _init(self):
        
        self.dynamics = DJIPhantomDynamics()
        self.state = np.zeros(12)
        self.tprev = 0
        self.t = 0

    def _render_hud(self, mode):
        
        from gym_copter.envs.rendering.hud import HUD

        if self.hud is None:
            self.hud = HUD()
 
        # Detect window close
        if not self.hud.isOpen(): return None

        return self.hud.display(mode, self.state)
