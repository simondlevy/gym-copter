'''
Pendulum class for Nengo adaptive controller

Copyright (C) 2021 Xuan Choo, Simon D. Levy

MIT License
'''

import nengo
import numpy as np
from adaptive import run


class Pendulum:
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

    def reset(self, seed):
        self.rng = np.random.RandomState(seed=seed)
        self.theta = self.rng.uniform(-self.limit, self.limit)
        self.dtheta = self.rng.uniform(-1, 1)

    def step(self, u):
        u = np.clip(u, -1, 1) * self.max_torque

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
        len0 = 40 * self.length
        x1 = 50
        y1 = 50
        x2 = x1 + len0 * np.sin(self.theta)
        y2 = y1 - len0 * np.cos(self.theta)
        x3 = x1 + len0 * np.sin(desired)
        y3 = y1 - len0 * np.cos(desired)
        return '''
        <svg width='100%' height='100%' viewbox='0 0 100 100'>
            <line x1='{x1}' y1='{y1}' x2='{x3}' y2='{y3}' style='stroke:blue'/>
            <line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}'
            style='stroke:black'/>
        </svg>
        '''.format(
            x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3
        )


with nengo.Network(seed=3) as model:

    net = run(Pendulum, 'Pendulum', 'Angle', 'Extra Force')
