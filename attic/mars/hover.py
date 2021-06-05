'''
Superclass for 2D and 3D copter hover

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from pidcontrollers import AltitudeHoldPidController
from task import _Task


class _Hover(_Task):

    BOUNDS = 10
    MAX_STEPS = 1000

    def __init__(self, observation_size, action_size):

        _Task.__init__(self, observation_size, action_size)

        # Set up altitude-hold PID controller for heuristic demo
        self.altpid = AltitudeHoldPidController()

    def _get_reward(self, status, state, d, x, y):

        # Simple reward for each step we complete
        return 1
