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

        phi_targ = y*self.PID_A + dy*self.PID_B
        phi_todo = ((phi-phi_targ)*self.PID_C + phi*self.PID_D -
                    dphi*self.PID_E)

        theta_targ = x*self.PID_A + dx*self.PID_B
        theta_todo = ((-theta-theta_targ)*self.PID_C - theta*self.PID_D +
                      dtheta*self.PID_E)

        hover_todo = z*self.PID_F + dz*self.PID_G

        # map throttle demand from [-1,+1] to [0,1]
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
