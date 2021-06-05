'''
Superclass for 2D and 3D copter lander

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np

from gym_copter.pidcontrollers import DescentPidController
from gym_copter.envs.task import _Task


class _Lander(_Task):

    TARGET_RADIUS = 2
    YAW_PENALTY_FACTOR = 50
    XYZ_PENALTY_FACTOR = 25
    DZ_MAX = 10
    DZ_PENALTY = 100

    INSIDE_RADIUS_BONUS = 100

    def __init__(self, observation_size, action_size):

        _Task.__init__(self, observation_size, action_size)

        # Add PID controller for heuristic demo
        self.descent_pid = DescentPidController()

    def _get_reward(self, status, state, d, x, y):

        # Get penalty based on state and motors
        shaping = -(self.XYZ_PENALTY_FACTOR*np.sqrt(np.sum(state[0:6]**2)) +
                    self.YAW_PENALTY_FACTOR*np.sqrt(np.sum(state[10:12]**2)))

        if (abs(state[d.STATE_Z_DOT]) > self.DZ_MAX):
            shaping -= self.DZ_PENALTY

        reward = ((shaping - self.prev_shaping)
                  if (self.prev_shaping is not None)
                  else 0)

        self.prev_shaping = shaping

        if status == d.STATUS_LANDED:

            self.done = True
            self.spinning = False

            # Win bigly we land safely between the flags
            if np.sqrt(x**2+y**2) < self.TARGET_RADIUS:

                reward += self.INSIDE_RADIUS_BONUS

        return reward
