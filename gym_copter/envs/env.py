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

    def __init__(self, dt=0.001, disp='hud'):

        self.num_envs = 1
        self.display = None

        # We handle time differently if we're rendering
        self.dt = dt

        # Default to HUD display
        self.disp = disp

        # Also called by reset()
        self._init()

    def _update(self, action):

        # Update dynamics and get kinematic state
        self.dynamics.setMotors(action)
        self.dynamics.update(self.dt)
        self.state = self.dynamics.getState()

        # Update timestep
        self.tick += 1

        # We're done when vehicle has crashed
        self.done = self.state[4]>0 and self.state[5]>1

        return self.done

    def reset(self):

        self._init()
        return self.state

    def render(self, mode='human'):

        # Track time
        tcurr = time()
        self.dt = (tcurr - self.tprev) if self.tprev > 0 else self.dt
        self.tprev = tcurr

        if self.display is None:
            self.display = HUD()
 
        return self.display.display(mode, self.state) if self.display.isOpen() else None

    def render3d(self, mode):

        from gym_copter.envs.rendering.tpv import TPV

        if self.display is None:
            self.display = TPV(self.unwrapped.spec.id)
 
        return self.display.display(mode, self.state) if self.display.isOpen() else None

    def close(self):

        Env.close(self)        

    def time(self):

        return self.tick * self.dt

    def _init(self):
        
        self.state = np.zeros(12)
        self.dynamics = DJIPhantomDynamics()
        self.tprev = 0
        self.tick = 0
        self.done = False

