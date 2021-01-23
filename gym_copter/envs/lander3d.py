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

    OBSERVATION_SIZE = 12
    ACTION_SIZE = 4

    # Parameters to adjust
    PITCH_ROLL_PENALTY_FACTOR = 0  # 250
    YAW_PENALTY_FACTOR = 50
    ZDOT_PENALTY_FACTOR = 10
    MOTOR_PENALTY_FACTOR = 0.03
    RESTING_DURATION = 1.0  # render a short while after successful landing
    LANDING_BONUS = 100
    XYZ_PENALTY_FACTOR = 25   # designed so that maximal penalty is around 100

    def __init__(self):

        Lander.__init__(self)

        # Pre-convert max-angle degrees to radian
        self.max_angle = np.radians(self.MAX_ANGLE)

    def reset(self):

        return Lander._reset(self, self._perturb())

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
            motors = np.clip(action, 0, 1)    # stay in interval [0,1]
            d.setMotors(self._get_motors(motors))
            self.spinning = sum(motors) > 0
            d.update()

        # Get new state from dynamics
        state = np.array(d.getState())

        # Extract components from state
        x, dx, y, dy, z, dz, phi, dphi, theta, dtheta, psi, dpsi = state

        # Set pose for display
        self.pose = x, y, z, phi, theta, psi

        # Get penalty based on state and motors
        shaping = -self._get_penalty(state, motors)

        reward = ((shaping - self.prev_shaping)
                  if (self.prev_shaping is not None)
                  else 0)
        self.prev_shaping = shaping

        # Assume we're not done yet
        done = False

        # Lose bigly if we go out of bounds
        if abs(x) >= self.BOUNDS or abs(y) >= self.BOUNDS:
            done = True
            reward = -self.OUT_OF_BOUNDS_PENALTY

        # Lose bigly for excess roll or pitch
        if abs(phi) >= self.max_angle or abs(theta) >= self.max_angle:
            done = True
            reward = -self.OUT_OF_BOUNDS_PENALTY

        # It's all over once we're on the ground
        if status in (d.STATUS_LANDED, d.STATUS_CRASHED):

            done = True

        # Bonus for good landing
        if status == d.STATUS_LANDED:

            reward += self.LANDING_BONUS

        return (np.array(self._get_state(state),
                dtype=np.float32),
                reward,
                done,
                {})

    def render(self, mode='human'):
        '''
        Returns None because we run viewer on a separate thread
        '''
        return None

    def _get_penalty(self, state, motors):

        return (self.XYZ_PENALTY_FACTOR*np.sqrt(np.sum(state[0:6]**2)) +
                self.PITCH_ROLL_PENALTY_FACTOR *
                np.sqrt(np.sum(state[6:10]**2)) +
                self.YAW_PENALTY_FACTOR * np.sqrt(np.sum(state[10:12]**2)) +
                self.MOTOR_PENALTY_FACTOR * np.sum(motors))

    def _get_bonus(self, x, y):

        return (self.INSIDE_RADIUS_BONUS
                if x**2+y**2 < self.LANDING_RADIUS**2
                else 0)

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
