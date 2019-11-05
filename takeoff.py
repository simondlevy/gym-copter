#!/usr/bin/env python3
'''
Test simple altitude-hold PID controller

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from time import sleep
import numpy as np

from gym_copter.dynamics.quadxap import QuadXAPDynamics
from gym_copter.dynamics import Parameters

# Target 
ALTITUDE_TARGET = 10

# PID params
ALT_P = 1.0
VEL_P = 1.0
VEL_I = 0
VEL_D = 0

# Time constant
DT = 0.001

class AltitudePidController(object):

    def __init__(self, target, posP, velP, velI, velD, windupMax=10):

        # In a real PID controller, this would be a set-point
        self.target = target

        # Constants
        self.posP = posP
        self.velP = velP
        self.velI = velI
        self.velD = velD
        self.windupMax = windupMax

        # Values modified in-flight
        self.posTarget      = 0
        self.lastError      = 0
        self.integralError  = 0

    def u(self, alt, vel, dt):

        # Compute dzdt setpoint and error
        velTarget = (self.target - alt) * self.posP
        velError = velTarget - vel

        # Update error integral and error derivative
        self.integralError +=  velError * dt
        self.integralError = AltitudePidController._constrainAbs(self.integralError + velError * dt, self.windupMax)
        deltaError = (velError - self.lastError) / dt if abs(self.lastError) > 0 else 0
        self.lastError = velError

        # Compute control u
        return self.velP * velError + self.velD * deltaError + self.velI * self.integralError

    def _constrainAbs(x, lim):

        return -lim if x < -lim else (+lim if x > +lim else x)

if __name__ == '__main__':

    # Dynamical parameters for copter
    params = Parameters(

            # Estimated
            5.E-06, # b
            2.E-06, # d

            # https:#www.dji.com/phantom-4/info
            1.380,  # m (kg)
            0.350,  # l (meters)

            # Estimated
            2,      # Ix
            2,      # Iy
            3,      # Iz
            38E-04, # Jr
            15000)  # maxrpm

    copter = QuadXAPDynamics(params)

    # initial conditions
    z = 0
    zprev = 0

    # Create PID controller
    pid  = AltitudePidController(ALTITUDE_TARGET, ALT_P, VEL_P, VEL_I, VEL_D)

    # Loop until user hits the stop button
    while True:

        # Extract altitude from state.  Altitude is in NED coordinates, so we negate it to use as input
        # to PID controller.
        z = -copter.getState().pose.location[2]

        print('%+3.3f' % z)

        # Use temporal first difference to compute vertical velocity
        dzdt = (z-zprev) / DT

        # Get correction from PID controller
        u = pid.u(z, dzdt, DT)

        # Constrain correction to [0,1] to represent motor value
        u = max(0, min(1, u))

        # Set motor values in sim
        copter.setMotors(u*np.ones(4))

        # Update the dynamics
        copter.update(DT)

        # Update for first difference
        zprev = z

        # Yield to Multicopter thread
        sleep(.001)
