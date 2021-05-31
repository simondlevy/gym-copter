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
    YAW_PENALTY_FACTOR = 50
    XYZ_PENALTY_FACTOR = 25
    DZ_MAX = 10
    DZ_PENALTY = 100

    INSIDE_RADIUS_BONUS = 100
    BOUNDS = 10

    def __init__(self, vehicle_name, observation_size=10):

        _Task.__init__(self, observation_size, 4, vehicle_name)

        # Add PID controller for heuristic demo
        self.descent_pid = DescentPidController()

        # Pre-convert max-angle degrees to radians
        self.max_angle = np.radians(self.MAX_ANGLE)

        # For generating CSV file
        self.STATE_NAMES = ['X', 'dX', 'Y', 'dY', 'Z', 'dZ',
                            'Phi', 'dPhi', 'Theta', 'dTheta']

        # Add PID controllers for heuristic demo
        self.phi_rate_pid = AngularVelocityPidController()
        self.theta_rate_pid = AngularVelocityPidController()
        self.x_poshold_pid = PositionHoldPidController()
        self.y_poshold_pid = PositionHoldPidController()

    def reset(self):

        return _Task._reset(self)

    def render(self, mode='human'):
        '''
        Returns None because we run viewer on a separate thread
        '''
        return None

    def demo_pose(self, args):

        x, y, z, phi, theta, viewer = args

        while viewer.is_open():

            self._reset(pose=(x, y, z, phi, theta), perturb=False)

            self.render()

            sleep(.01)

        self.close()

    def heuristic(self, state, nopid):
        '''
        PID controller
        '''
        x, dx, y, dy, z, dz, phi, dphi, theta, dtheta = state

        phi_todo = 0
        theta_todo = 0

        if not nopid:

            phi_rate_todo = self.phi_rate_pid.getDemand(dphi)
            y_pos_todo = self.x_poshold_pid.getDemand(y, dy)
            phi_todo = phi_rate_todo + y_pos_todo

            theta_rate_todo = self.theta_rate_pid.getDemand(-dtheta)
            x_pos_todo = self.y_poshold_pid.getDemand(x, dx)
            theta_todo = theta_rate_todo + x_pos_todo

        descent_todo = self.descent_pid.getDemand(z, dz)

        t, r, p = (descent_todo+1)/2, phi_todo, theta_todo

        return [t-r-p, t+r+p, t+r-p, t-r+p]  # use mixer to set motors

    def _get_motors(self, motors):

        return motors

    def _get_state(self, state):

        return state[:10]


def make_parser():
    '''
    Exported function to support command-line parsing in scripts.
    You can add your own arguments, then call parse() to get args.
    '''

    parser = _make_parser()

    # Add 3D-specific argument support

    parser.add_argument('--view', required=False, default='30,120',
                        help='Elevation, azimuth for view perspective')

    group = parser.add_mutually_exclusive_group()

    group.add_argument('--vision', action='store_true',
                       help='Use vision sensor')

    group.add_argument('--dvs', action='store_true',
                       help='Use Dynamic Vision Sensor')

    group.add_argument('--nodisplay', action='store_true',
                       help='Suppress display')

    return parser


def parse(parser):
    args = parser.parse_args()
    viewangles = tuple((int(s) for s in args.view.split(',')))
    return args, viewangles


def main():

    parser = make_parser()

    parser.add_argument('--freeze', dest='pose', required=False,
                        default=None, help='Freeze in pose x,y,z,phi,theta')

    args, viewangles = parse(parser)

    env = Drop3D(args.vehicle)

    if not args.nodisplay:
        viewer = ThreeDLanderRenderer(env, viewangles=viewangles)

    threadfun = env.demo_heuristic
    threadargs = args.seed, args.nopid, args.csvfilename

    if args.pose is not None:
        try:
            x, y, z, phi, theta = (float(s) for s in args.pose.split(','))
        except Exception:
            print('POSE must be x,y,z,phi,theta')
            exit(1)
        threadfun = env.demo_pose
        threadargs = (x, y, z, phi, theta, viewer)

    thread = threading.Thread(target=threadfun, args=threadargs)

    thread.start()

    if not args.nodisplay:
        viewer.start()


if __name__ == '__main__':

    main()
