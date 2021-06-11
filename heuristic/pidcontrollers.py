'''
PID controllers for heuristic demos

Copyright (C) 2021 Simon D. Levy

MIT License
'''


class AltitudeHoldPidController:

    def __init__(self, Kp=0.2, Ki=3, target=5, windupMax=0.2):

        self.Kp = Kp
        self.Ki = Ki

        self.target = target

        # Prevents integral windup
        self.windupMax = windupMax

        # Error integral
        self.errorI = 0

    def getDemand(self, z, dz):

        # Negate for NED => ENU
        z, dz = -z, -dz

        # Velocity is a setpoint
        targetVelocity = self.target - z

        # Compute error as scaled target minus actual
        error = targetVelocity - dz

        # Compute P term
        pterm = error * self.Kp

        # Compute I term
        self.errorI += error

        # avoid integral windup
        self.errorI = AltitudeHoldPidController.constrain(self.errorI,
                                                          self.windupMax)
        iterm = self.errorI * self.Ki

        return pterm + iterm

    @staticmethod
    def constrain(val, lim):
        return -lim if val < -lim else (+lim if val > +lim else val)


class PositionHoldPidController:

    def __init__(self, Kd=4):

        self.Kd = Kd

        # Accumulated values
        self.lastError = 0

    def getDemand(self, x, dx):

        error = -x - dx

        dterm = (error - self.lastError) * self.Kd

        self.lastError = error

        return dterm


class DescentPidController:

    def __init__(self, Kp=1.15, Kd=1.33):

        self.Kp = Kp
        self.Kd = Kd

    def getDemand(self, z, dz):

        return z*self.Kp + dz*self.Kd


class AngularVelocityPidController:

    def getDemand(self, angularVelocity):

        return -angularVelocity
