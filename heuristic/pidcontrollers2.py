'''
PID controllers for heuristic demos

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np


class _PidController:

    def __init__(self, Kp, Kd):

        self.Kp = Kp
        self.Kd = Kd

        # Accumulated values
        self.lastError = 0
        self.deltaError1 = 0
        self.deltaError2 = 0

        # For deltaT-based controllers
        self.previousTime = 0

    def compute(self, target, actual, debug=False):

        # Compute error as scaled target minus actual
        error = target - actual

        # Compute P term
        pterm = error * self.Kp

        # Compute D term
        dterm = 0
        if self.Kd > 0:  # optimization
            deltaError = error - self.lastError
            dterm = ((self.deltaError1 + self.deltaError2 + deltaError)
                     * self.Kd)
            self.deltaError2 = self.deltaError1
            self.deltaError1 = deltaError
            self.lastError = error

        return pterm + dterm


class PositionHoldPidController:

    def __init__(self, Kd=4):

        self.velPid = _PidController(0, Kd)

    def getDemand(self, x, dx):

        # Velocity is a setpoint
        targetVelocity = -x

        # Run velocity PID controller to get correction
        return self.velPid.compute(targetVelocity, dx)
