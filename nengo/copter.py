'''
Quadcopter class for Nengo adaptive controller

Copyright (C) 2021 Xuan Choo, Simon D. Levy

MIT License
'''

import nengo
import gym

from adaptive import run


def _constrain(val, lim):
    return -lim if val < -lim else (+lim if val > +lim else val)


class _AltitudeHoldPidController:

    def __init__(self, k_p=0.2, k_i=3, k_tgt=5, k_windup=0.2):

        self.k_p = k_p
        self.k_i = k_i

        self.k_tgt = k_tgt

        # Prevents integral windup
        self.k_windup = k_windup

        # Error integral
        self.ei = 0

    def getDemand(self, z, dz):

        # Negate for NED => ENU
        z, dz = -z, -dz

        # Compute error as scaled target minus actual
        e = (self.k_tgt - z) - dz

        # Compute I term
        self.ei += e

        # avoid integral windup
        self.ei = _constrain(self.ei, self.k_windup)

        return e * self.k_p + self.ei * self.k_i


class Copter:

    def __init__(self, seed=None):

        self.env = gym.make('gym_copter:Hover2D-v0')
        self.reset(seed)

    def reset(self, seed):

        self.state = self.env.reset()

        self.alt_pid = _AltitudeHoldPidController()

    def step(self, u):

        self.env.render()

        _y, _dy, z, dz, _phi, _dphi = self.state

        hover_todo = self.alt_pid.getDemand(z, dz)

        self.state, _reward, _done, _ = self.env.step([hover_todo, hover_todo])

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
