#!/usr/bin/env python3
'''
3D Copter-Lander class

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np
import threading

from gym_copter.envs.lander import Lander

from gym_copter.rendering.threed import ThreeDLanderRenderer
from gym_copter.rendering.threed import ThreeDLanderRendererVisual
from gym_copter.rendering.threed import make_parser, parse


class Lander3D(Lander):

    # 3D model
    OBSERVATION_SIZE = 10
    ACTION_SIZE = 4

    # Number of target points
    TARGET_POINTS = 250

    # Angle PID for heuristic demo
    PID_C = 0.025

    def __init__(self):

        Lander.__init__(self)

        # Pre-convert max-angle degrees to radian
        self.max_angle = np.radians(self.MAX_ANGLE)

        # Create points for landing zone
        pts = np.linspace(-np.pi, +np.pi, self.TARGET_POINTS)
        self.target = np.array([self.LANDING_RADIUS * np.sin(pts),
                                self.LANDING_RADIUS * np.cos(pts)])

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

    def _get_motors(self, motors):

        return motors

    def _get_state(self, state):

        return state[:10]

    def heuristic(self, state):

        x, dx, y, dy, z, dz, phi, dphi, theta, dtheta = state

        phi_todo = self._angle_pid(y, dy, phi, dphi)

        theta_todo = self._angle_pid(x, dx, -theta, -dtheta)

        hover_todo = self._hover_pid(z, dz)

        t, r, p = (hover_todo+1)/2, phi_todo, theta_todo

        return [t-r-p, t+r+p, t+r-p, t-r+p]  # use mixer to set motors


class Lander3DVisual(Lander3D):

    # Arbitrary specs of a hypothetical low-resolution camera with square
    # sensor.
    RESOLUTION = 128  # pixels
    FIELD_OF_VIEW = 60  # degrees
    SENSOR_SIZE = .008  # meters

    def __init__(self):

        Lander3D.__init__(self)

        # http://paulbourke.net/miscellaneous/lens/
        self.focal_length = (0.5 * self.SENSOR_SIZE /
                             np.tan(np.radians(self.FIELD_OF_VIEW/2)))

    def get_target_points(self):

        # XXX Transform the target by perspective projection
        target = self.target.copy()

        return target

# End of Lander3D classes -------------------------------------------------


def main():

    parser = make_parser()
    args, viewangles = parse(parser)

    if args.visual:
        env = Lander3DVisual()
        viewer = ThreeDLanderRendererVisual(env, viewangles=viewangles)
    else:
        env = Lander3D()
        viewer = ThreeDLanderRenderer(env, viewangles=viewangles)

    thread = threading.Thread(target=env.demo_heuristic, args=(args.seed, ))
    thread.start()
    viewer.start()


if __name__ == '__main__':

    main()
