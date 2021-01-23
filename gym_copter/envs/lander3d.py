#!/usr/bin/env python3
'''
3D Copter-Lander super-class (no ground target)

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import time
import numpy as np
import threading

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from gym_copter.dynamics.djiphantom import DJIPhantomDynamics
from gym_copter.rendering.threed import ThreeDLanderRenderer
from gym_copter.rendering.threed import make_parser, parse


class Lander3D(gym.Env, EzPickle):

    # Parameters to adjust
    INITIAL_RANDOM_FORCE = 0  # perturbation for initial position
    INITIAL_ALTITUDE = 5
    PITCH_ROLL_PENALTY_FACTOR = 0  # 250
    YAW_PENALTY_FACTOR = 50
    ZDOT_PENALTY_FACTOR = 10
    MOTOR_PENALTY_FACTOR = 0.03
    BOUNDS = 10
    OUT_OF_BOUNDS_PENALTY = 100
    RESTING_DURATION = 1.0  # render a short while after successful landing
    FRAMES_PER_SECOND = 50
    MAX_ANGLE = 45   # big penalty if roll or pitch angles go beyond this
    EXCESS_ANGLE_PENALTY = 100
    LANDING_BONUS = 100
    LANDING_RADIUS = 2
    INSIDE_RADIUS_BONUS = 100
    INITIAL_RANDOM_FORCE = 30  # perturbation for initial position
    XYZ_PENALTY_FACTOR = 25   # designed so that maximal penalty is around 100


    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FRAMES_PER_SECOND
    }

    def __init__(self):

        EzPickle.__init__(self)
        self.seed()

        self.prev_reward = None

        # Observation is all state values
        self.observation_space = (
                spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32))

        # Action is four floats (one per motor)
        self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)

        # Support for rendering
        self.viewer = None
        self.pose = None

        # Pre-convert max-angle degrees to radian
        self.max_angle = np.radians(self.MAX_ANGLE)

        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.prev_shaping = None

        # Create cusom dynamics model
        self.dynamics = DJIPhantomDynamics(self.FRAMES_PER_SECOND)

        # Initialize custom dynamics with random perturbations
        state = np.zeros(12)
        d = self.dynamics
        state[d.STATE_X] = 0
        state[d.STATE_Y] = 0
        state[d.STATE_Z] = -self.INITIAL_ALTITUDE
        self.dynamics.setState(state)
        self.dynamics.perturb(np.array([self._perturb(),  # X
                                        self._perturb(),  # Y
                                        self._perturb(),  # Z
                                        0,                # phi
                                        0,                # theta
                                        0]))              # psi

        return self.step(np.array([0, 0, 0, 0]))[0]

    def step(self, action):

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
            reward += self.LANDING_BONUS

        # Support Novelty Search
        info = {'behavior': behavior}

        return np.array(state, dtype=np.float32), reward, done, info

    def render(self, mode='human'):
        '''
        Returns None because we run viewer on a separate thread
        '''
        return None

    def close(self):
        return

    def get_radius(self):

        # XXX should come from a superclass
        return 2.0

    def get_pose(self):

        return self.pose


    def _perturb(self):

        return np.random.uniform(-self.INITIAL_RANDOM_FORCE,
                                 + self.INITIAL_RANDOM_FORCE)

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

    def _get_state(self, state):

        return state[self.dynamics.STATE_Z:len(state)]

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

        return [t-r-p, t+r+p, t+r-p, t-r+p]  # use mixer to set motors# End of Lander3D classes ----------------------------------------------------


def heuristic_lander(env, heuristic, viewer=None, seed=None):

    if seed is not None:
        env.seed(seed)
        np.random.seed(seed)

    total_reward = 0
    steps = 0
    state = env.reset()

    while True:

        action = heuristic(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if steps % 20 == 0 or done:
            print('observations:',
                  ' '.join(['{:+0.2f}'.format(x) for x in state]))
            print('step {} total_reward {:+0.2f}'.format(steps, total_reward))

        steps += 1

        if done:
            break

        if viewer is not None:
            time.sleep(1./env.FRAMES_PER_SECOND)

    env.close()
    return total_reward


def demo(env):

    parser = make_parser()

    args, viewangles = parse(parser)

    renderer = ThreeDLanderRenderer(env, viewangles=viewangles)

    thread = threading.Thread(target=heuristic_lander,
                              args=(env, env.heuristic, renderer))
    thread.start()

    renderer.start()


if __name__ == '__main__':

    demo(Lander3D())
