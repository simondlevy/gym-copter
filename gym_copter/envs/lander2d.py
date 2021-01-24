#!/usr/bin/env python3
'''
2D Copter-Lander, based on
  https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py

This version controls each motor separately

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import argparse
from argparse import ArgumentDefaultsHelpFormatter
import numpy as np
from gym_copter.envs.lander import Lander


class Lander2D(Lander):

    # 3D model
    OBSERVATION_SIZE = 6
    ACTION_SIZE = 2

    PENALTY_FACTOR = 12

    def __init__(self):

        Lander.__init__(self)

    def reset(self):

        if self.viewer is not None:
            self.viewer.close()

        return Lander._reset(self)

    def step(self, action):

        # Abbreviation
        d = self.dynamics
        status = d.getStatus()

        motors = np.zeros(4)

        # Stop motors after safe landing
        if status == d.STATUS_LANDED:
            d.setMotors(motors)
            self.spinning = False

        # In air, set motors from action
        else:
            motors = np.clip(action, 0, 1)    # keep motors in interval [0,1]
            d.setMotors([motors[0], motors[1], motors[1], motors[0]])
            self.spinning = sum(motors) > 0
            d.update()

        # Get new state from dynamics
        _, _, posy, vely, posz, velz, phi, velphi = d.getState()[:8]

        # Set lander pose for viewer
        self.pose = posy, posz, phi

        # Convert state to usable form
        state = np.array([posy, vely, posz, velz, phi, velphi])

        # Get penalty based on state and motors
        shaping = -self._get_penalty(state, motors)

        reward = ((shaping - self.prev_shaping)
                  if (self.prev_shaping is not None)
                  else 0)

        self.prev_shaping = shaping

        # Assume we're not done yet
        done = False

        # Lose bigly if we go outside window
        if abs(posy) >= self.BOUNDS:
            done = True
            reward -= self.OUT_OF_BOUNDS_PENALTY

        else:

            # It's all over once we're on the ground
            if status == d.STATUS_LANDED:

                done = True
                self.spinning = False

                # Win bigly we land safely between the flags
                if abs(posy) < self.LANDING_RADIUS:

                    reward += self.INSIDE_RADIUS_BONUS

            elif status == d.STATUS_CRASHED:

                # Crashed!
                done = True
                self.spinning = False

        return np.array(state, dtype=np.float32), reward, done, {}

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
        A = 0.1
        B = 0.1

        # Angle PID
        C = 0.1
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
        return self.PENALTY_FACTOR * np.sqrt(np.sum(state[0:4]**2))


def main():
    parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, required=False, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    Lander2D().demo_heuristic(seed=args.seed)


if __name__ == '__main__':
    main()
