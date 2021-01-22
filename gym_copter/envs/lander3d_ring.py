#!/usr/bin/env python3
'''
3D Copter-Lander with ring (radius) target

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from gym_copter.envs.lander3d import demo
from gym_copter.envs.lander3d_point import Lander3DPoint


class Lander3DRing(Lander3DPoint):

    LANDING_RADIUS = 2
    INSIDE_RADIUS_BONUS = 100

    def __init__(self):

        Lander3DPoint.__init__(self)

    def _get_bonus(self, x, y):

        return (self.INSIDE_RADIUS_BONUS
                if x**2+y**2 < self.LANDING_RADIUS**2
                else 0)

    def get_radius(self):

        return self.LANDING_RADIUS


if __name__ == '__main__':

    demo(Lander3DRing())
