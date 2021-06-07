'''
Superclass for 2D and 3D copter hover

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from gym_copter.envs.task import _Task


class _Hover(_Task):

    def __init__(self, observation_size, action_size):

        _Task.__init__(self, observation_size, action_size)

    def _get_reward(self, status, state, d, x, y):

        # Simple reward for each step we complete
        return 1
