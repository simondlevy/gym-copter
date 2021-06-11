'''
PID controllers for heuristic demos

Copyright (C) 2021 Simon D. Levy

MIT License
'''


class _PiController:

    def __init__(self, Kp, Ki, windup_max=0.2):

        self.Kp = Kp
        self.Ki = Ki

        # Prevents integral windup
        self.windupMax = windup_max

        # Error integral
        self.errorI = 0

    def compute(self, target, actual):

        # Compute error as scaled target minus actual
        error = target - actual

        # Compute P term
        pterm = error * self.Kp

        # Compute I term
        iterm = 0
        if self.Ki > 0:  # optimization

            self.errorI += error

            # avoid integral windup
            self.errorI = _PiController.constrain(self.errorI, self.windupMax)
            iterm = self.errorI * self.Ki

        return pterm + iterm

    def reset(self):

        self.errorI = 0

    @staticmethod
    def constrain(val, lim):
        return -lim if val < -lim else (+lim if val > +lim else val)


class AltitudeHoldPidController:

    def __init__(self, Kp=0.2, Ki=3, target=5):

        self.posPid = _PiController(1, 0, 0)
        self.velPid = _PiController(Kp, Ki, 0)

        self.target = target

    def getDemand(self, z, dz):

        # Velocity is a setpoint (negated for NED => ENU)
        targetVelocity = self.posPid.compute(self.target, -z)

        # Run velocity PID controller to get correction (negate for NED => ENU)
        return self.velPid.compute(targetVelocity, -dz)


class PositionHoldPidController:

    def __init__(self, Kd=4):

        self.Kd = Kd

        # Accumulated values
        self.lastError = 0
        self.deltaError1 = 0
        self.deltaError2 = 0

    def getDemand(self, x, dx):

        error = -x - dx
        deltaError = error - self.lastError
        dterm = ((self.deltaError1 + self.deltaError2 + deltaError) * self.Kd)
        self.deltaError2 = self.deltaError1
        self.deltaError1 = deltaError
        self.lastError = error

        return dterm

##############################################################################


class DescentPidController:

    def __init__(self, Kp=1.15, Kd=1.33):

        self.Kp = Kp
        self.Kd = Kd

    def getDemand(self, z, dz):

        return z*self.Kp + dz*self.Kd


class AngularVelocityPidController:

    def getDemand(self, angularVelocity):

        return -angularVelocity
