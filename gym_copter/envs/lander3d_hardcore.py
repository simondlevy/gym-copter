#!/usr/bin/env python3
'''
3D Copter-Lander with full dynamics (12 state values)

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym_copter.envs.lander3d import Lander3D, run


class Lander3DHardcore(Lander3D):

    LANDING_RADIUS = 2
    INSIDE_RADIUS_BONUS = 100

    def __init__(self):

        Lander3D.__init__(self)

    def _get_bonus(self, x, y):

        return (self.INSIDE_RADIUS_BONUS
                if x**2+y**2 < self.LANDING_RADIUS**2
                else 0)

    def step_novelty(self, action):
        return self._step(action)


if __name__ == '__main__':

    run(Lander3DHardcore(), 2)
