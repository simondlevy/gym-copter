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

    def seed(self, seed=None):

        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        # Support for rendering
        self.pose = None
        self.spinning = False
        self.prev_shaping = None

        # Create cusom dynamics model
        self.dynamics = DJIPhantomDynamics(self.FRAMES_PER_SECOND)

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

        print("steps =  {} total_reward {:+0.2f}".format(steps, total_reward))

        sleep(1)
        self.close()
        return total_reward
