#!/usr/bin/env python3
'''
3D Copter-Lander class

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from time import sleep
import numpy as np
import threading
import argparse
from argparse import ArgumentDefaultsHelpFormatter

from gym_copter.envs.lander import Lander

from gym_copter.rendering.threed import ThreeDLanderRenderer
from gym_copter.rendering.threed import ThreeDVisualLanderRenderer

from gym_copter.dynamics.djiphantom import DJIPhantomDynamics


class Lander3D(Lander):

    # 3D model
    OBSERVATION_SIZE = 10
    ACTION_SIZE = 4

    # Number of target points (arbitrary)
    TARGET_POINTS = 250

    # Angle PID for heuristic demo
    PID_C = 0.025

    def __init__(self):

        Lander.__init__(self)

        # Pre-convert max-angle degrees to radian
        self.max_angle = np.radians(self.MAX_ANGLE)

        # Create points for landing zone
        self.target = self._get_target_points()

    def reset(self):

        return Lander._reset(self)

    def render(self, mode='human'):
        '''
        Returns None because we run viewer on a separate thread
        '''
        return None

    def done(self):

        d = self.dynamics

        return d.getStatus() != d.STATUS_AIRBORNE

    def demo_pose(self, args):

        # self._reset()

        # Support for rendering
        self.pose = None
        self.spinning = False
        self.prev_shaping = None

        # Create dynamics model
        self.dynamics = DJIPhantomDynamics(self.FRAMES_PER_SECOND)

        # Set up initial conditions
        state = np.zeros(12)
        d = self.dynamics
        state[d.STATE_X] = 0
        state[d.STATE_Y] = 0
        state[d.STATE_Z] = -5
        self.dynamics.setState(state)

        # Step to initial state
        self.step(np.zeros(self.ACTION_SIZE))[0]

        x, y, z, phi, theta, viewer = args

        while viewer.is_open():

            self.render()

            sleep(.01)

        self.close()

    def _get_target_points(self, scale=1.0):

        pts = np.linspace(-np.pi, +np.pi, self.TARGET_POINTS)

        return scale * np.array([np.sin(pts), np.cos(pts)])

    def _get_motors(self, motors):

        return motors

    def _get_state(self, state):

        return state[:10]

    def heuristic(self, state):
        '''
        PID controller
        '''
        x, dx, y, dy, z, dz, phi, dphi, theta, dtheta = state
        phi_todo = self._angle_pid(y, dy, phi, dphi)
        theta_todo = self._angle_pid(x, dx, -theta, -dtheta)
        hover_todo = self._hover_pid(z, dz)
        t, r, p = (hover_todo+1)/2, phi_todo, theta_todo
        return [t-r-p, t+r+p, t+r-p, t-r+p]  # use mixer to set motors


class Lander3DVisual(Lander3D):

    # Arbitrary specs of a hypothetical low-resolution camera with square
    # sensor
    # RESOLUTION = 128  # pixels
    FIELD_OF_VIEW = 60  # degrees
    SENSOR_SIZE = .008  # meters

    def __init__(self):

        Lander3D.__init__(self)

        # Get focal length f from equations in
        # http://paulbourke.net/miscellaneous/lens/
        #
        # field of view = 2 atan(0.5 width / focallength)
        #
        # Therefore focalllength = width / (2 tan(field of view /2))
        #
        self.f = (self.SENSOR_SIZE /
                  (2 * np.tan(np.radians(self.FIELD_OF_VIEW/2))))

    def get_target_image_points(self):

        # Get distance u to image as negated NED altitude
        u = -np.array(self.dynamics.getState())[4]

        # Get image magnification m from equations in
        # https://www.aplustopper.com/numerical-methods-in-lens/
        #
        # 1/u + 1/v = 1/f
        #
        # m = v/u
        #
        # Therefore m = 1 / (u/f - 1)
        #
        m = 1 / (u / self.f - 1)

        return self._get_target_points(scale=m)


# End of Lander3D classes -------------------------------------------------

def make_parser():
    '''
    Exported function to support command-line parsing in scripts.
    You can add your own arguments, then call parse() to get args.
    '''
    parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--view', required=False, default='30,120',
                        help='View elevation, azimuth')
    parser.add_argument('--seed', type=int, required=False, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--visual', action='store_true',
                        help='Run visual environment')
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

    if args.visual:
        env = Lander3DVisual()
        viewer = ThreeDVisualLanderRenderer(env, viewangles=viewangles)
    else:
        env = Lander3D()
        viewer = ThreeDLanderRenderer(env, viewangles=viewangles)

    threadfun = env.demo_heuristic
    threadargs = args.seed

    if args.pose is not None:
        try:
            x, y, z, phi, theta = (int(s) for s in args.pose.split(','))
        except Exception:
            print('POSE must be x,y,z,phi,theta')
            exit(1)
        threadfun = env.demo_pose
        threadargs = (x, y, z, phi, theta, viewer)

    thread = threading.Thread(target=threadfun, args=(threadargs, ))

    thread.start()
    viewer.start()


if __name__ == '__main__':

    main()
