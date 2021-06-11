'''
PID controllers for heuristic demos

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np


class _PController:

    def __init__(self, Kp):

        self.Kp = Kp

    def compute(self, target, actual):

        # Compute error as scaled target minus actual
        error = target - actual

        # Compute P term
        return error * self.Kp

class AngularVelocityPidController(_PController):

    def __init__(self, Kp=1.0):

        _PController.__init__(self, Kp)

    def getDemand(self, angularVelocity):

        return _PController.compute(self, 0, angularVelocity)


class PositionHoldPidController:

    def __init__(self, Kd=0.1):

        self.Kd = Kd

    def getDemand(self, _x, dx):

        return -dx * self.Kd
