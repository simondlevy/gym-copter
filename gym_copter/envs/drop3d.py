#!/usr/bin/env python3
'''
Class for testing gravity by dropping

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from time import sleep
import numpy as np
import threading

from gym_copter.envs.utils import _make_parser
from gym_copter.rendering.threed import ThreeDLanderRenderer
from gym_copter.pidcontrollers import AngularVelocityPidController
from gym_copter.pidcontrollers import PositionHoldPidController

from gym_copter.pidcontrollers import DescentPidController
from gym_copter.envs.task import _Task


class Drop3D(_Task):

    TARGET_RADIUS = 2

    def __init__(self, vehicle_name, observation_size=10):

        _Task.__init__(self, observation_size, 4, vehicle_name)

        # For generating CSV file
        self.STATE_NAMES = ['X', 'dX', 'Y', 'dY', 'Z', 'dZ',
                            'Phi', 'dPhi', 'Theta', 'dTheta']

    def reset(self):

        return _Task._reset(self)

    def render(self, mode='human'):
        '''
        Returns None because we run viewer on a separate thread
        '''
        return None

    def heuristic(self, state, nopid):

        return [0, 0, 0, 0]

    def _get_motors(self, motors):

        return motors

    def _get_state(self, state):

        return state[:10]


def main():

    parser = _make_parser()

    # Add 3D-specific argument support

    parser.add_argument('--view', required=False, default='30,120',
                        help='Elevation, azimuth for view perspective')

    parser.add_argument('--nodisplay', action='store_true',
                        help='Suppress display')

    parser.add_argument('--freeze', dest='pose', required=False,
                        default=None, help='Freeze in pose x,y,z,phi,theta')

    args = parser.parse_args()
    viewangles = tuple((int(s) for s in args.view.split(',')))

    env = Drop3D(args.vehicle)

    if not args.nodisplay:
        viewer = ThreeDLanderRenderer(env, viewangles=viewangles)

    threadfun = env.demo_heuristic
    threadargs = args.seed, args.nopid, args.csvfilename
    thread = threading.Thread(target=threadfun, args=threadargs)

    thread.start()

    if not args.nodisplay:
        viewer.start()


if __name__ == '__main__':

    main()
