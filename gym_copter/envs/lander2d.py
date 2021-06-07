#!/usr/bin/env python3
'''
2D Copter lander

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym_copter.envs.parsing import _make_parser
from gym_copter.envs.lander import _Lander
from gym_copter.pidcontrollers import AngularVelocityPidController
from gym_copter.pidcontrollers import PositionHoldPidController


class Lander2D(_Lander):

    def __init__(self):

        _Lander.__init__(self, 6, 2)

        # Add PID controllers for heuristic demo
        self.rate_pid = AngularVelocityPidController()
        self.poshold_pid = PositionHoldPidController()

        # For generating CSV file
        self.STATE_NAMES = ['X', 'dX', 'Z', 'dZ', 'Phi', 'dPhi']

    def reset(self):

        if self.viewer is not None:
            self.viewer.close()

        return _Lander._reset(self)

    def render(self, mode='human'):

        print(str(self) + ' render')

        from gym_copter.rendering.twod import TwoDLanderRenderer

        # Create viewer if not done yet
        if self.viewer is None:
            self.viewer = TwoDLanderRenderer(self)

        return None if self.steps%2 else self.viewer.render(mode, self.pose, self.spinning)

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

        hover_todo = self.descent_pid.getDemand(z, dz)

        return hover_todo-phi_todo, hover_todo+phi_todo


def main():
    parser = _make_parser()
    args = parser.parse_args()
    Lander2D().demo_heuristic(seed=args.seed,
                              nopid=args.nopid,
                              csvfilename=args.csvfilename)


if __name__ == '__main__':
    main()
