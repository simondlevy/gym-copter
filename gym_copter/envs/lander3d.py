#!/usr/bin/env python3
'''
3D Copter-Lander super-class (no ground target)

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np
import threading

from gym_copter.envs.lander import Lander

from gym_copter.rendering.threed import ThreeDLanderRenderer
from gym_copter.rendering.threed import make_parser, parse


class Lander3D(Lander):

    # 3D model
    OBSERVATION_SIZE = 12
    ACTION_SIZE = 4

    def __init__(self):

        Lander.__init__(self)

        # Pre-convert max-angle degrees to radian
        self.max_angle = np.radians(self.MAX_ANGLE)

    def reset(self):

        return Lander._reset(self, self._perturb())

    def render(self, mode='human'):
        '''
        Returns None because we run viewer on a separate thread
        '''
        return None

    def _get_motors(self, motors):

        return motors

    def _get_state(self, state):

        return state

    def heuristic(self, s):

        x, dx, y, dy, z, dz, phi, dphi, theta, dtheta = s[:10]

        phi_todo = self._angle_pid(y, dy, phi, dphi)

        theta_todo = self._angle_pid(x, dx, -theta, -dtheta)

        hover_todo = self._hover_pid(z, dz)

        t, r, p = (hover_todo+1)/2, phi_todo, theta_todo

        return [t-r-p, t+r+p, t+r-p, t-r+p]  # use mixer to set motors

    # End of Lander3D classes -------------------------------------------------


def demo(env):

    parser = make_parser()
    args, viewangles = parse(parser)
    renderer = ThreeDLanderRenderer(env, viewangles=viewangles)
    thread = threading.Thread(target=env.demo_heuristic)
    thread.start()
    renderer.start()


if __name__ == '__main__':

    demo(Lander3D())
