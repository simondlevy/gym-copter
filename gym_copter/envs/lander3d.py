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
from gym_copter.sensors.vision import VisionSensor


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
        pts = np.linspace(-np.pi, +np.pi, self.TARGET_POINTS)
        self.target = np.array([np.sin(pts), np.cos(pts)]).transpose()

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

        x, y, z, phi, theta, viewer = args

        self._reset(pose=(x, y, z, phi, theta), perturb=False)

        while viewer.is_open():

            self.render()

            sleep(.01)

        self.close()

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

    RESOLUTION = 128

    def __init__(self):

        Lander3D.__init__(self)

        self.vision_sensor = VisionSensor(resolution=self.RESOLUTION)

    def get_target_image_points(self):

        # Extract pose
        x, y, z, phi, theta, _ = self.pose

        # Start with original target
        target = self.target.copy()

        # Add x,y offset
        target[:, 0] += x
        target[:, 1] += y

        # XXX need to skew by phi, theta

        return target

    def get_target_image(self):

        # Extract pose
        x, y, z, phi, theta, _ = self.pose

        # Negate NED altitude to get distance to target
        return self.vision_sensor.get_image(self.get_target_image_points(), -z)


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
