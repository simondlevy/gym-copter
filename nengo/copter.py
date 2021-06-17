'''
Quadcopter class for Nengo adaptive controller

Copyright (C) 2021 Xuan Choo, Simon D. Levy

MIT License
'''

import nengo
import gym

from adaptive import run


class Copter:

    def __init__(self, seed=None):

        self.env = gym.make('gym_copter:Hover1D-v0')
        self.reset(seed)

    def reset(self, seed):

        self.state = self.env.reset()

    def step(self, u):

        self.env.render()

        z, dz, = self.state

        self.state, _reward, _done, _ = self.env.step((0,0))

        return 0, 0

    def set_extra_force(self, force):

        self.extra_mass = force

    def generate_html(self, desired):
        '''
        Copter is simulated externally
        '''
        return None


with nengo.Network(seed=3) as model:

    run(Copter, 'Copter', 'Position', 'Wind Force')
