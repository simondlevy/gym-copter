'''
PID controllers for heuristic demos

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np


class AngularVelocityPidController:

    def getDemand(self, angularVelocity):

        return -angularVelocity


class PositionHoldPidController:

    def __init__(self, Kd=0.1):

        self.Kd = Kd

    def getDemand(self, _x, dx):

        return -dx * self.Kd
