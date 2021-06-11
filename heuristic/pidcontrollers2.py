'''
PID controllers for heuristic demos

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np


class _DController:

    def __init__(self, Kd):

        self.Kd = Kd

        # Accumulated values
        self.lastError = 0
        self.deltaError1 = 0
        self.deltaError2 = 0

    def compute(self, target, actual):

        # Compute error as scaled target minus actual
        error = target - actual

        deltaError = error - self.lastError
        dterm = ((self.deltaError1 + self.deltaError2 + deltaError) * self.Kd)
        self.deltaError2 = self.deltaError1
        self.deltaError1 = deltaError
        self.lastError = error

        return dterm


class PositionHoldPidController:

    def __init__(self, Kd=4):

        self.velPid = _DController(Kd)

    def getDemand(self, x, dx):

        # Run velocity PID controller to get correction
        return self.velPid.compute(-x, dx)
