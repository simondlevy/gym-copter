#!/usr/bin/env python3
'''
2D Copter hovering

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from gym_copter.envs.utils import _make_parser
from gym_copter.envs.hover import _Hover
from gym_copter.pidcontrollers import AngularVelocityPidController
from gym_copter.pidcontrollers import PositionHoldPidController


class Hover2D(_Hover):

    def __init__(self):

        _Hover.__init__(self, 6, 2)

        # Add PID controllers for heuristic demo
        self.rate_pid = AngularVelocityPidController()
        self.poshold_pid = PositionHoldPidController()

        # For generating CSV file
        self.STATE_NAMES = ['X', 'dX', 'Z', 'dZ', 'Phi', 'dPhi']

    def reset(self):

        if self.viewer is not None:
            self.viewer.close()

        return _Hover._reset(self)

    def render(self, mode='human'):

        from gym_copter.rendering.twod import TwoDRenderer

        # Create viewer if not done yet
        if self.viewer is None:
            self.viewer = TwoDRenderer()

        self.viewer.render(mode, self.pose, self.spinning)

        return self.viewer._complete(mode)

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
            pos_todo = self.poshold_pid.getDemand(y, dy)

            phi_todo = rate_todo + pos_todo

        hover_todo = self.altpid.getDemand(z, dz)

        return hover_todo-phi_todo, hover_todo+phi_todo


def main():
    parser = _make_parser()
    args = parser.parse_args()
    Hover2D().demo_heuristic(seed=args.seed,
                             nopid=args.nopid,
                             csvfilename=args.csvfilename)


if __name__ == '__main__':
    main()