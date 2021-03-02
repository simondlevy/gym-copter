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

    def __init__(self, Kp_pos=1.0, Kp_vel=0.2, Ki_vel=3, Kd_vel=0, target=5):

        self.posPid = _PidController(Kp_pos, 0, 0)
        self.velPid = _PidController(Kp_vel, Ki_vel, Kd_vel)

        self.target = target

    def getDemand(self, z, dz):

        # Negate for NED
        z, dz = -z, -dz

        # Target velocity is a setpoint
        targetVelocity = self.posPid.compute(self.target, z)

        # Run velocity PID controller to get correction
        return self.velPid.compute(targetVelocity, dz)


class PositionHoldPidController:

    def __init__(self,
                 Kp_pos=1.0, Kp_vel=0.00001, Ki_vel=0.1, Kd_vel=1,
                 target=0):

        self.posPid = _PidController(Kp_pos, 0, 0)
        self.velPid = _PidController(Kp_vel, Ki_vel, Kd_vel)

        self.target = target

    def getDemand(self, x, dx):

        # Target velocity is a setpoint
        targetVelocity = self.posPid.compute(self.target, x, debug=True)

        # Run velocity PID controller to get correction
        correction = self.velPid.compute(targetVelocity, dx)

        return correction


class TargetPidController:

    def __init__(self, X_Kp=0.1, X_Kd=0.1, Target_Kp=0.1,
                 Phi_Kp=0.05, Phi_Kd=0.4):

        self.X_Kp = X_Kp
        self.X_Kd = X_Kd
        self.Target_Kp = Target_Kp
        self.Phi_Kp = Phi_Kp
        self.Phi_Kd = Phi_Kd

    def getDemand(self, x, dx, phi, dphi):

        phi_targ = x*self.X_Kp + dx*self.X_Kd

        return ((phi-phi_targ)*self.Target_Kp
                + phi*self.Phi_Kp - dphi*self.Phi_Kd)


class DescentPidController:

    def __init__(self, Kp=1.15, Kd=1.33):

        self.kP = Kp
        self.kd = Kd

        _PidController.__init__(self, Kp, 0, Kd)

    def getDemand(self, z, dz):

        return z*self.Kp + dz*self.Kd


class AnglePidController(_PidController):

    def __init__(self, Kp=0):

        _PidController.__init__(self, Kp, 0, 0)

    def getDemand(self, angle):

        return _PidController.compute(self, 0, angle)


class AngularVelocityPidController(_PidController):

    # Arbitrary constants
    BIG_DEGREES_PER_SECOND = 40
    WINDUP_MAX = 6

    def __init__(self, Kp=0, Ki=0, Kd=0):

        _PidController.__init__(self, Kp, Ki, Kd, self.WINDUP_MAX)

        # Convert degree parameters to radians for use later
        self.bigAngularVelocity = np.radians(self.BIG_DEGREES_PER_SECOND)

    def getDemand(self, angularVelocity):

        # Reset integral on quick angular velocity change
        if abs(angularVelocity) > self.bigAngularVelocity:
            self.reset()

        return _PidController.compute(self, 0, angularVelocity)
