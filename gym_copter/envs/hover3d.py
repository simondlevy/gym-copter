#!/usr/bin/env python3
'''
3D Copter-Hover class

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from time import sleep
from numpy import degrees, radians
import threading

from gym_copter.envs.utils import _make_parser
from gym_copter.envs.hover import _Hover
from gym_copter.rendering.threed import ThreeDHoverRenderer
from gym_copter.sensors.vision.vs import VisionSensor
from gym_copter.sensors.vision.dvs import DVS
from gym_copter.pidcontrollers import AngularVelocityPidController
from gym_copter.pidcontrollers import PositionHoldPidController


class Hover3D(_Hover):

    def __init__(self, vehicle_name, obs_size=12):

        _Hover.__init__(self, obs_size, 4, vehicle_name)

        # Pre-convert max-angle degrees to radians
        self.max_angle = radians(self.MAX_ANGLE)

        # For generating CSV file
        self.STATE_NAMES = ['X', 'dX', 'Y', 'dY', 'Z', 'dZ',
                            'Phi', 'dPhi', 'Theta', 'dTheta', 'Psi', 'dPsi']

        # Add PID controllers for heuristic demo
        self.roll_rate_pid = AngularVelocityPidController()
        self.pitch_rate_pid = AngularVelocityPidController()
        self.yaw_rate_pid = AngularVelocityPidController()
        self.x_poshold_pid = PositionHoldPidController()
        self.y_poshold_pid = PositionHoldPidController()

    def reset(self):

        return _Hover._reset(self)

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
        x, dx, y, dy, z, dz, phi, dphi, theta, dtheta, _, dpsi = state

        roll_todo = 0
        pitch_todo = 0
        yaw_todo = 0

        if not nopid:

            roll_rate_todo = self.roll_rate_pid.getDemand(dphi)
            y_pos_todo = self.x_poshold_pid.getDemand(y, dy)

            pitch_rate_todo = self.pitch_rate_pid.getDemand(-dtheta)
            x_pos_todo = self.y_poshold_pid.getDemand(x, dx)

            roll_todo = roll_rate_todo + y_pos_todo
            pitch_todo = pitch_rate_todo + x_pos_todo
            yaw_todo = self.yaw_rate_pid.getDemand(-dpsi)

        hover_todo = self.altpid.getDemand(z, dz)

        t, r, p, y = (hover_todo+1)/2, roll_todo, pitch_todo, yaw_todo

        # Use mixer to set motors
        return [t-r-p-y, t+r+p-y, t+r-p+y, t-r+p+y]

    def _get_motors(self, motors):

        return motors

    def _get_state(self, state):

        return state


class HoverVisual(Hover3D):

    RES = 16

    def __init__(self, vs=VisionSensor(res=RES)):

        Hover3D.__init__(self)

        self.vs = vs

        self.image = None

    def step(self, action):

        result = Hover3D.step(self, action)

        x, y, z, phi, theta, psi = self.pose

        self.image = self.vs.getImage(x,
                                      y,
                                      max(-z, 1e-6),  # keep Z positive
                                      degrees(phi),
                                      degrees(theta),
                                      degrees(psi))

        return result

    def render(self, mode='human'):

        if self.image is not None:
            self.vs.display_image(self.image)


class HoverDVS(HoverVisual):

    def __init__(self):

        HoverVisual.__init__(self, vs=DVS(res=HoverVisual.RES))

# End of Hover3D classes -------------------------------------------------


def make_parser():
    '''
    Exported function to support command-line parsing in scripts.
    You can add your own arguments, then call parse() to get args.
    '''

    # Start with general-purpose parser from _Hover superclass
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

    env = (HoverDVS(args.vehicle_name) if args.dvs
           else (HoverVisual() if args.vision
                 else Hover3D(args.vehicle_name)))

    if not args.nodisplay:
        viewer = ThreeDHoverRenderer(env, viewangles=viewangles)

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
