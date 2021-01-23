#!/usr/bin/env python3
'''
2D Copter-Lander, based on
  https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py

This version controls each motor separately

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import argparse
from argparse import ArgumentDefaultsHelpFormatter
from gym_copter.envs.lander import Lander


class Lander2D(Lander):

    # 2D model
    OBSERVATION_SIZE = 6
    ACTION_SIZE = 2

    def __init__(self):

        Lander.__init__(self)

    def reset(self):

        if self.viewer is not None:
            self.viewer.close()

        return Lander._reset(self, 0)

    def render(self, mode='human'):

        from gym_copter.rendering.twod import TwoDLanderRenderer

        # Create viewer if not done yet
        if self.viewer is None:
            self.viewer = TwoDLanderRenderer(self.LANDING_RADIUS)

        return self.viewer.render(mode, self.pose, self.spinning)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def heuristic(self, s):

        y, dy, z, dz, phi, dphi = s

        phi_todo = self._angle_pid(y, dy, phi, dphi)

        hover_todo = self._hover_pid(z, dz)

        return hover_todo-phi_todo, hover_todo+phi_todo

    def _get_motors(self, motors):

        return [motors[0], motors[1], motors[1], motors[0]]

    def _get_state(self, state):

        return state[2:8]


def main():
    parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, required=False, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    Lander2D().demo_heuristic(seed=args.seed)


if __name__ == '__main__':
    main()
