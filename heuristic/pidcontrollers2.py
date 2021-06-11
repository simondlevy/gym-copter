'''
PID controllers for heuristic demos

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np


class PositionHoldPidController:

    def __init__(self, Kd=4):

        self.Kd = Kd

        # Accumulated values
        self.lastError = 0
        self.deltaError1 = 0
        self.deltaError2 = 0

    def getDemand(self, x, dx):

        target = -x
        actual = dx

        # Compute error as scaled target minus actual
        error = target - actual

        deltaError = error - self.lastError
        dterm = ((self.deltaError1 + self.deltaError2 + deltaError) * self.Kd)
        self.deltaError2 = self.deltaError1
        self.deltaError1 = deltaError
        self.lastError = error

        return dterm
