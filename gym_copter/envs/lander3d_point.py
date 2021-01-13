#!/usr/bin/env python3
'''
3D Copter-Lander with point target

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np

from gym import spaces
from gym.utils import seeding, EzPickle

from gym_copter.envs.lander3d import Lander3D, demo
from gym_copter.dynamics.djiphantom import DJIPhantomDynamics


class Lander3DPoint(Lander3D):

    # Parameters to adjust
    XY_PENALTY_FACTOR = 25   # designed so that maximal penalty is around 100

    def __init__(self):

        Lander3D.__init__(self)

        # Observation is all state values
        self.observation_space = (
                spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32))

    def reset(self):

        self.prev_shaping = None

        # Create cusom dynamics model
        self.dynamics = DJIPhantomDynamics(self.FRAMES_PER_SECOND)

        # Initialize custom dynamics with random perturbations
        state = np.zeros(12)
        d = self.dynamics
        state[d.STATE_X], state[d.STATE_Y] = self._get_initial_offset()
        state[d.STATE_Z] = -self.INITIAL_ALTITUDE
        self.dynamics.setState(state)

        return self.step(np.array([0, 0, 0, 0]))[0]

    def step(self, action):

        state, reward, _, done, info = self._step(action)

        return state, reward, done, info

    def get_radius(self):

        return 0.1

    def _get_initial_offset(self):

        # return 2.5 * np.random.randn(2)
        return 4, 4

    def _step(self, action):
        '''
        Takes an action and returns a tuple
        (state, reward, behavior, done, info).
        Behavior is tuple (x, y, vertical velocity).
        '''

        # Abbreviation
        d = self.dynamics

        # Get current status (landed, crashed, ...)
        status = d.getStatus()

        # Keep motors in interval [0,1]
        motors = (np.zeros(4)
                  if status == d.STATUS_LANDED
                  else np.clip(action, 0, 1))

        d.setMotors(motors)

        # Update dynamics if airborne
        if status != d.STATUS_LANDED:
            d.update()

        # Get new state from dynamics
        state = np.array(d.getState())

        # Extract pose from state
        x, y, z, phi, theta, psi = state[0::2]

        # Set pose pose for display
        self.pose = x, y, z, phi, theta, psi

        # Reward is a simple penalty for overall distance and angle and their
        # first derivatives, plus a bit more for running motors (discourage
        # endless hover)
        shaping = -(
                self.XY_PENALTY_FACTOR*np.sqrt(np.sum(state[0:6]**2)) +
                self.PITCH_ROLL_PENALTY_FACTOR *
                np.sqrt(np.sum(state[6:10]**2)) +
                self.YAW_PENALTY_FACTOR * np.sqrt(np.sum(state[10:12]**2)) +
                self.MOTOR_PENALTY_FACTOR * np.sum(motors)
                )

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

        # No behavior until we've crashed or landed
        behavior = None

        # It's all over once we're on the ground
        if status in (d.STATUS_LANDED, d.STATUS_CRASHED):

            # Once we're one the ground, our behavior is our x,y position and
            # vertical velocity
            behavior = x, y, state[d.STATE_Z_DOT]

            done = True

        # Bonus for good landing
        if status == d.STATUS_LANDED:

            # Different subclasses add different bonuses for proximity to
            # center
            reward += self._get_bonus(x, y)

        return np.array(state, dtype=np.float32), reward, behavior, done, {}

    def _get_bonus(self, x, y):

        # Bonus is proximity to center
        return self.BOUNDS - np.sqrt(x**2+y**2)

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


# End of Lander3DPoint classes ------------------------------------------------

if __name__ == '__main__':

    demo(Lander3DPoint())
