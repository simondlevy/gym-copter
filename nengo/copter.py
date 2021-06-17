'''
Quadcopter class for Nengo adaptive controller

Copyright (C) 2021 Xuan Choo, Simon D. Levy

MIT License
'''

import nengo
import numpy as np
import gym

from adaptive import run


class Copter:

    def __init__(
        self,
        mass=4.0,
        length=1.0,
        dt=0.001,
        g=10.0,
        seed=None,
        max_torque=100,
        max_speed=8,
        limit=2.0,
        bounds=None,
    ):
        self.mass = mass
        self.length = length
        self.dt = dt
        self.g = g
        self.max_torque = max_torque
        self.max_speed = max_speed
        self.limit = limit
        self.extra_mass = 0
        self.bounds = bounds
        self.reset(seed)

        self.env = gym.make('gym_copter:Hover1D-v0')

    def reset(self, seed):

        self.rng = np.random.RandomState(seed=seed)
        self.theta = self.rng.uniform(-self.limit, self.limit)
        self.dtheta = self.rng.uniform(-1, 1)

    def step(self, u):

        u = np.clip(u, -1, 1) * self.max_torque

        print(u)

        mass = self.mass + self.extra_mass
        self.dtheta += (
            -3 * self.g / (2 * self.length) * np.sin(self.theta + np.pi)
            + 3.0 / (mass * self.length ** 2) * u
        ) * self.dt
        self.theta += self.dtheta * self.dt
        self.dtheta = np.clip(self.dtheta, -self.max_speed, self.max_speed)

        if self.bounds:
            self.theta = np.clip(self.theta, self.bounds[0], self.bounds[1])
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        return self.theta, self.dtheta

    def set_extra_force(self, force):

        self.extra_mass = force

    def generate_html(self, desired):
        '''
        Copter is simulated externally
        '''
        return None


with nengo.Network(seed=3) as model:

    run(Copter, 'Copter', 'Position', 'Wind Force')
