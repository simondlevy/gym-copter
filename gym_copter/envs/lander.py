'''
Superclass for 2D and 3D copter lander

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np
from time import sleep
import gym
from gym import spaces
from gym.utils import EzPickle, seeding

from gym_copter.dynamics.djiphantom import DJIPhantomDynamics


class Lander(gym.Env, EzPickle):

    # Parameters to adjust
    INITIAL_RANDOM_FORCE = 30  # perturbation for initial position
    INITIAL_ALTITUDE = 10
    LANDING_RADIUS = 2
    BOUNDS = 10
    OUT_OF_BOUNDS_PENALTY = 100
    FRAMES_PER_SECOND = 50
    INSIDE_RADIUS_BONUS = 100
    MAX_ANGLE = 45   # big penalty if roll or pitch angles go beyond this

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FRAMES_PER_SECOND
    }

    def __init__(self):

        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        self.pose = None
        self.prev_reward = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf,
                                            +np.inf,
                                            shape=(self.OBSERVATION_SIZE,),
                                            dtype=np.float32)

        # Action is two floats [throttle, roll]
        self.action_space = spaces.Box(-1,
                                       +1,
                                       (self.ACTION_SIZE,),
                                       dtype=np.float32)

        # Pre-convert max-angle degrees to radians
        self.max_angle = np.radians(self.MAX_ANGLE)

    def seed(self, seed=None):

        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self, xperturb):

        # Support for rendering
        self.pose = None
        self.spinning = False
        self.prev_shaping = None

        # Create dynamics model
        self.dynamics = DJIPhantomDynamics(self.FRAMES_PER_SECOND)

        # Set up initial conditions
        state = np.zeros(12)
        d = self.dynamics
        state[d.STATE_X] = 0
        state[d.STATE_Y] = 0
        state[d.STATE_Z] = -self.INITIAL_ALTITUDE
        self.dynamics.setState(state)

        # Perturb with a random force
        self.dynamics.perturb(np.array([xperturb,          # X
                                        self._perturb(),   # Y
                                        self._perturb(),   # Z
                                        0,                 # phi
                                        0,                 # theta
                                        0]))               # psi

        # Return initial state
        return self.step(np.zeros(self.ACTION_SIZE))[0]

    def demo_heuristic(self, seed=None, render=True):

        self.seed(seed)
        np.random.seed(seed)
        self.seed(seed)

        total_reward = 0
        steps = 0
        state = self.reset()

        while True:

            action = self.heuristic(state)
            state, reward, done, _ = self.step(action)
            total_reward += reward

            self.render('rgb_array')

            sleep(1./self.FRAMES_PER_SECOND)

            steps += 1

            if done:
                break

        print('steps =  %d total_reward = %+0.2f' % (steps, total_reward))

        sleep(1)
        self.close()
        return total_reward

    def _perturb(self):

        return np.random.uniform(-self.INITIAL_RANDOM_FORCE,
                                 + self.INITIAL_RANDOM_FORCE)
