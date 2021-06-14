'''
PID controllers for heuristic demos

Copyright (C) 2021 Simon D. Levy

MIT License
'''


def _constrain(val, lim):
    return -lim if val < -lim else (+lim if val > +lim else val)


class AltitudeHoldPidController:

    def __init__(self, k_p=0.2, k_i=3, k_targ=5, k_windup=0.2):

        self.k_p = k_p
        self.k_i = k_i

        self.k_targ = k_targ

        # Prevents integral windup
        self.k_windup = k_windup

        # Error integral
        self.ei = 0

    def getDemand(self, z, dz):

        # Negate for NED => ENU
        z, dz = -z, -dz

        # Velocity is a setpoint
        dz_targ = self.k_targ - z

        # Compute error as scaled target minus actual
        e = dz_targ - dz

        # Compute I term
        self.ei += e

        # avoid integral windup
        self.ei = _constrain(self.ei, self.k_windup)

        return e * self.k_p + self.ei * self.k_i


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

    def __init__(self, k_p=1.15, Kd=1.33):

        self.k_p = k_p
        self.Kd = Kd

    def getDemand(self, z, dz):

        return z*self.k_p + dz*self.Kd


class AngularVelocityPidController:

    def getDemand(self, angularVelocity):

        return -angularVelocity
