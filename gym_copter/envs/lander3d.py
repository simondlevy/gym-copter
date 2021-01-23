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

    def _get_penalty(self, state, motors):

        return (self.XYZ_PENALTY_FACTOR*np.sqrt(np.sum(state[0:6]**2)) +
                self.YAW_PENALTY_FACTOR * np.sqrt(np.sum(state[10:12]**2)) +
                self.MOTOR_PENALTY_FACTOR * np.sum(motors))

    def _get_motors(self, motors):

        return motors

    def _get_state(self, state):

        return state

    @staticmethod
    def heuristic(s):
        '''
        The heuristic for
        1. Testing
        2. Demonstration rollout.

        Args:
            s (list): The state. Attributes:
                      s[0] is the X coordinate
                      s[1] is the X speed
                      s[2] is the Y coordinate
                      s[3] is the Y speed
                      s[4] is the vertical coordinate
                      s[5] is the vertical speed
                      s[6] is the roll angle
                      s[7] is the roll angular speed
                      s[8] is the pitch angle
                      s[9] is the pitch angular speed
         returns:
             a: The heuristic to be fed into the step function defined above to
                determine the next step and reward.  '''

        # Angle target
        A = 0.05
        B = 0.06

        # Angle PID
        C = 0.025
        D = 0.05
        E = 0.4

        # Vertical PID
        F = 1.15
        G = 1.33

        x, dx, y, dy, z, dz, phi, dphi, theta, dtheta = s[:10]

        phi_targ = y*A + dy*B              # angle should point towards center
        phi_todo = (phi-phi_targ)*C + phi*D - dphi*E

        theta_targ = x*A + dx*B         # angle should point towards center
        theta_todo = -(theta+theta_targ)*C - theta*D + dtheta*E

        hover_todo = z*F + dz*G

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
