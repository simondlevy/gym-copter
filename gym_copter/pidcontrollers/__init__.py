'''
PID controllers for heuristic demos

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np


class _PidController:

    def __init__(self, Kp, Ki, Kd, windup_max=0.2):

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        # Prevents integral windup
        self.windupMax = windup_max

        # Accumulated values
        self.lastError = 0
        self.errorI = 0
        self.deltaError1 = 0
        self.deltaError2 = 0

        # For deltaT-based controllers
        self.previousTime = 0

    def compute(self, target, actual, debug=False):

        # Compute error as scaled target minus actual
        error = target - actual

        # Compute P term
        pterm = error * self.Kp

        # Compute I term
        iterm = 0
        if self.Ki > 0:  # optimization

            # avoid integral windup
            self.errorI = _PidController.constrainAbs(self.errorI + error,
                                                      self.windupMax)
            iterm = self.errorI * self.Ki

        # Compute D term
        dterm = 0
        if self.Kd > 0:  # optimization
            deltaError = error - self.lastError
            dterm = ((self.deltaError1 + self.deltaError2 + deltaError)
                     * self.Kd)
            self.deltaError2 = self.deltaError1
            self.deltaError1 = deltaError
            self.lastError = error

        return pterm + iterm + dterm

    def reset(self):

        self.errorI = 0
        self.lastError = 0
        self.previousTime = 0

    @staticmethod
    def constrainMinMax(val, minval, maxval):
        return minval if val < minval else (maxval if val > maxval else val)

    @staticmethod
    def constrainAbs(val, maxval):
        return _PidController.constrainMinMax(val, -maxval, +maxval)


class AltitudeHoldPidController:

    def __init__(self, Kp=0.2, Ki=3, Kd=0, target=5):

        self.posPid = _PidController(1, 0, 0)
        self.velPid = _PidController(Kp, Ki, Kd)

        self.target = target

    def getDemand(self, z, dz):

        # Negate for NED
        z, dz = -z, -dz

        # Velocity is a setpoint
        targetVelocity = self.posPid.compute(self.target, z)

        # Run velocity PID controller to get correction
        return self.velPid.compute(targetVelocity, dz)


class PositionHoldPidController:

    def __init__(self, Kp=0.00001, Ki=0.1, Kd=4, target=0):

        self.posPid = _PidController(1, 0, 0)
        self.velPid = _PidController(Kp, Ki, Kd)

        self.target = target

    def getDemand(self, x, dx):

        # Velocity is a setpoint
        targetVelocity = self.posPid.compute(self.target, x)

        # Run velocity PID controller to get correction
        return self.velPid.compute(targetVelocity, dx)


class DescentPidController:

    def __init__(self, Kp=1.15, Kd=1.33):

        self.kP = Kp
        self.kd = Kd

        _PidController.__init__(self, Kp, 0, Kd)

    def getDemand(self, z, dz):

        return z*self.Kp + dz*self.Kd


class AngularVelocityPidController(_PidController):

    # Arbitrary constants
    BIG_DEGREES_PER_SECOND = 40
    WINDUP_MAX = 6

    def __init__(self, Kp=1.0, Ki=0, Kd=1):

        _PidController.__init__(self, Kp, Ki, Kd, self.WINDUP_MAX)

        # Convert degree parameters to radians for use later
        self.bigAngularVelocity = np.radians(self.BIG_DEGREES_PER_SECOND)

    def getDemand(self, angularVelocity):

        # Reset integral on quick angular velocity change
        if abs(angularVelocity) > self.bigAngularVelocity:
            self.reset()

        return _PidController.compute(self, 0, angularVelocity)
