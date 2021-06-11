'''
PID controllers for heuristic demos

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np


class DescentPidController:

    def __init__(self, Kp=1.15, Kd=1.33):

        self.Kp = Kp
        self.Kd = Kd

    def getDemand(self, z, dz):

        return z*self.Kp + dz*self.Kd


class AngularVelocityPidController:

    def getDemand(self, angularVelocity):

        return -angularVelocity
