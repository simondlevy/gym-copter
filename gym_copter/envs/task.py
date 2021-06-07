'''
Abstract class for 2D and 3D copter environments

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import abc

import numpy as np
from numpy import radians
from time import time

import gym
from gym import spaces
from gym.utils import EzPickle, seeding

from gym_copter.dynamics import Dynamics, djiphantom_params


class _Task(gym.Env, EzPickle):

    FRAMES_PER_SECOND = 100

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FRAMES_PER_SECOND
    }

    def __init__(self, observation_size, action_size,
                 initial_random_force=30,
                 out_of_bounds_penalty=100,
                 max_steps=1000,
                 max_angle=45,
                 bounds=10,
                 initial_altitude=10):

        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        self.pose = None
        self.action_size = action_size

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf,
                                            +np.inf,
                                            shape=(observation_size,),
                                            dtype=np.float32)

        # Action is two floats [throttle, roll]
        self.action_space = spaces.Box(-1,
                                       +1,
                                       (action_size,),
                                       dtype=np.float32)

        # Pre-convert max-angle degrees to radians
        self.max_angle = np.radians(max_angle)

        # Grab remaining settings
        self.initial_random_force = initial_random_force
        self.out_of_bounds_penalty = out_of_bounds_penalty
        self.max_steps = max_steps
        self.bounds = bounds
        self.initial_altitude = initial_altitude

    def seed(self, seed=None):

        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        # Abbreviation
        d = self.dynamics
        status = d.getStatus()

        motors = np.zeros(4)

        # Stop motors after safe landing
        if status == d.STATUS_LANDED:
            self.spinning = False

        # In air, set motors from action
        else:
            motors = np.clip(action, 0, 1)    # stay in interval [0,1]
            self.spinning = sum(motors) > 0
            d.update(self._get_motors(motors))

        # Get new state from dynamics
        state = np.array(d.getState())

        # Extract components from state
        x, dx, y, dy, z, dz, phi, dphi, theta, dtheta, psi, dpsi = state

        # Set pose for display
        self.pose = x, y, z, phi, theta, psi

        # Assume we're not done yet
        self.done = False

        reward = self._get_reward(status, state, d, x, y)

        # Lose bigly if we go outside window
        if abs(x) >= self.bounds or abs(y) >= self.bounds:
            self.done = True
            reward -= self.out_of_bounds_penalty

        # Lose bigly for excess roll or pitch
        elif abs(phi) >= self.max_angle or abs(theta) >= self.max_angle:
            self.done = True
            reward = -self.out_of_bounds_penalty

        # It's all over if we crash
        elif status == d.STATUS_CRASHED:

            # Crashed!
            self.done = True
            self.spinning = False

        # Don't run forever!
        if self.steps == self.max_steps:
            self.done = True
        self.steps += 1

        # Extract 2D or 3D components of state and rerturn them with the rest
        return (np.array(self._get_state(state), dtype=np.float32),
                reward,
                self.done,
                {})

    def close(self):

        gym.Env.close(self)
        self.viewer.close()

    def _reset(self, pose=None, perturb=True):

        if pose is None:
            pose = (0, 0, self.initial_altitude, 0, 0)

        # Support for rendering
        self.pose = None
        self.spinning = False
        self.done = False

        # Support for reward shaping
        self.prev_shaping = None

        # Create dynamics model
        self.dynamics = Dynamics(djiphantom_params, self.FRAMES_PER_SECOND)

        # Set up initial conditions
        state = np.zeros(12)
        d = self.dynamics
        state[d.STATE_X] = pose[0]
        state[d.STATE_Y] = pose[1]
        state[d.STATE_Z] = -pose[2]  # NED
        state[d.STATE_PHI] = radians(pose[3])
        state[d.STATE_THETA] = radians(pose[4])
        self.dynamics.setState(state)

        # Perturb with a random force
        if perturb:
            self.dynamics.perturb(np.array([self._randforce(),  # X
                                            self._randforce(),  # Y
                                            self._randforce(),  # Z
                                            0,                  # phi
                                            0,                  # theta
                                            0]))                # psi

        # No steps or reward yet
        self.steps = 0

        # Helps synchronize rendering to dynamics
        self.start = time()

        # Return initial state
        return self.step(np.zeros(self.action_size))[0]

    def _randforce(self):

        return np.random.uniform(-self.initial_random_force,
                                 + self.initial_random_force)

    @abc.abstractmethod
    def _get_reward(self, status, state, d, x, y):
        return 0
