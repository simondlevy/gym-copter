#!/usr/bin/env python3
'''
2D Copter-Lander, based on
  https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py

This version controls each motor separately

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import numpy as np

from gym_copter.envs.lander import Lander


class Lander2D(Lander):

    # 2D model
    OBSERVATION_SIZE = 6
    ACTION_SIZE = 2

    def __init__(self):

        Lander.__init__(self)

    def reset(self):

        if self.viewer is not None:
            self.viewer.close()

        return Lander._reset(self, 0)

    def render(self, mode='human'):

        from gym_copter.rendering.twod import TwoDLanderRenderer

        # Create viewer if not done yet
        if self.viewer is None:
            self.viewer = TwoDLanderRenderer(self.LANDING_RADIUS)

        return self.viewer.render(mode, self.pose, self.spinning)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def heuristic(self, s):
        """
        The heuristic for
        1. Testing
        2. Demonstration rollout.

        Args:
            s (list): The state. Attributes:
                      s[0] is the horizontal coordinate
                      s[1] is the horizontal speed
                      s[2] is the vertical coordinate
                      s[3] is the vertical speed
                      s[4] is the angle
                      s[5] is the angular speed
        returns:
             a: The heuristic to be fed into the step function defined above to
                determine the next step and reward.
        """

        # Angle target
        A = 0.1  # 0.05
        B = 0.1  # 0.06

        # Angle PID
        C = 0.1  # 0.025
        D = 0.05
        E = 0.4

        # Vertical PID
        F = 1.15
        G = 1.33

        posy, vely, posz, velz, phi, velphi = s

        phi_targ = posy*A + vely*B         # angle should point towards center
        phi_todo = (phi-phi_targ)*C + phi*D - velphi*E

        hover_todo = posz*F + velz*G

        return hover_todo-phi_todo, hover_todo+phi_todo

    def _get_penalty(self, state, motors):

        # Penalize distance from center and velocity
        return self.PENALTY_FACTOR * np.sqrt(np.sum(state[0:6]**2))

    def _get_motors(self, motors):

        return [motors[0], motors[1], motors[1], motors[0]]

    def _get_state(self, state):

        return state[2:8]


def main():
    Lander2D().demo_heuristic()


if __name__ == '__main__':
    main()
