'''
Quadcopter class for Nengo adaptive controller

Copyright (C) 2021 Xuan Choo, Simon D. Levy

MIT License
'''

import nengo
import gymnasium as gym
import numpy as np

from adaptive import run


class Copter:

    def __init__(self, seed=None):

        self.env = gym.make('gym_copter:Hover1D-v0')
        self.reset(seed)

    def reset(self, seed):

        self.state = self.env.reset()

    def step(self, u):

        u = np.clip(u, 0, 1)

        self.env.render()

        z, dz, = self.state

        # Negate for NED => ENU
        z, dz = -z, -dz

        print('%f | %+3.3f   %+3.3f' % (u, z, dz))

        self.state, _reward, _done, _info = self.env.step((u,))

        return z, dz

    def set_extra_force(self, force):

        self.extra_mass = force

    def generate_html(self, desired):
        '''
        Copter is simulated externally
        '''
        return None


with nengo.Network(seed=3) as model:

    run(Copter, 'Copter', 'Position', 'Wind Force')
