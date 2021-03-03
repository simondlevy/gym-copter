#!/usr/bin/env python3
'''
2D Copter lander

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import numpy as np

from gym_copter.envs.lander import _Lander, _make_parser
from gym_copter.pidcontrollers import AnglePidController
from gym_copter.pidcontrollers import AngularVelocityPidController
from gym_copter.pidcontrollers import PositionHoldPidController


class Lander2D(_Lander):

    def __init__(self):

        _Lander.__init__(self, 6, 2)

        # Add PID controllers for heuristic demo
        self.level_pid = AnglePidController()
        self.rate_pid = AngularVelocityPidController()
        self.poshold_pid = PositionHoldPidController()

        # For generating CSV file
        self.STATE_NAMES = ['X', 'dX', 'Z', 'dZ', 'Phi', 'dPhi']

    def reset(self):

        if self.viewer is not None:
            self.viewer.close()

        return _Lander._reset(self)

    def render(self, mode='human'):

        from gym_copter.rendering.twod import TwoDLanderRenderer

        # Create viewer if not done yet
        if self.viewer is None:
            self.viewer = TwoDLanderRenderer(self.TARGET_RADIUS)

        return self.viewer.render(mode, self.pose, self.spinning)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _get_state(self, state):

        return state[2:8]

    def _get_motors(self, motors):

        return [motors[0], motors[1], motors[1], motors[0]]

    def heuristic(self, s, nopid):

        y, dy, z, dz, phi, dphi = s

        phi_todo = 0

        if not nopid:

            rate_todo = self.rate_pid.getDemand(dphi)
            level_todo = self.level_pid.getDemand(dphi)
            pos_todo = self.poshold_pid.getDemand(y, dy)

            phi_todo = rate_todo + level_todo + pos_todo

        hover_todo = self.descent_pid.getDemand(z, dz)

        return hover_todo-phi_todo, hover_todo+phi_todo

    def _get_penalty(self, state, motors):

        # Penalize distance from center and velocity
        return self.PENALTY_FACTOR * np.sqrt(np.sum(state[0:4]**2))


def main():
    parser = _make_parser()
    args = parser.parse_args()
    Lander2D().demo_heuristic(seed=args.seed,
                              nopid=args.nopid,
                              csvfilename=args.csvfilename)


if __name__ == '__main__':
    main()
