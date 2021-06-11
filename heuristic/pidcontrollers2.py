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

    def getDemand(self, x, dx):

        error = -x - dx

        deltaError = error - self.lastError

        dterm = deltaError * self.Kd

        self.lastError = error

        return dterm
