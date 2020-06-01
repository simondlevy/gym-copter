'''
gym-copter Environment superclass

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym import Env
import numpy as np
from time import time

from gym_copter.dynamics.djiphantom import DJIPhantomDynamics

class CopterEnv(Env):

    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, dt=0.001, statedims=12):

        self.num_envs = 1
        self.display = None

        # We handle time differently if we're rendering
        self.dt = dt
        self.dt_realtime = dt
        self.tprev = 0

        # Support custom state representations
        self.statedims = statedims

        # Also called by reset()
        self._reset()

    def _update(self, action):

        # Update dynamics and get kinematic state
        self.dynamics.setMotors(action)
        self.dynamics.update(self.dt_realtime)
        self.state[:12] = self.dynamics.getState()
        self.pose = self.state[0:6:2]

        # Update timestep
        self.tick += 1

        # We're done when vehicle has crashed
        self.done = self.state[4]>0 and self.state[5]>1

        return self.done

    def reset(self):

        self._reset()
        return self.state

    def render(self, mode='human'):

        # Default to HUD display if start3d() wasn't called
        if self.display is None:
            from gym_copter.envs.rendering.hud import HUD
            self.display = HUD()
 
        # Track time
        tcurr = time()
        self.dt_realtime = (tcurr - self.tprev) if self.tprev > 0 else self.dt
        self.tprev = tcurr

        return self.display.display(mode, self.state, self.dt_realtime*self.tick) if self.display.isOpen() else None

    def tpvplotter(self, showtraj=False):

        from gym_copter.envs.rendering.tpv import TPV

        # Pass title to 3D display
        return TPV(self, showtraj=showtraj)

    def close(self):

        Env.close(self)        

    def time(self):

        return self.tick * self.dt_realtime

    def _reset(self):
        
        self.state = np.zeros(self.statedims)
        self.dynamics = DJIPhantomDynamics()
        self.tick = 0
        self.done = False
