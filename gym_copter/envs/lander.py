#!/usr/bin/env python3
'''
Superclass for 2D and 3D copter lander

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np
import gym
from gym import spaces
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
