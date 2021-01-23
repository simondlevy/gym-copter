#!/usr/bin/env python3
'''
Superclass for 2D and 3D copter lander

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np
import gym
from gym.utils import EzPickle, seeding


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
        self.prev_reward = None

    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
