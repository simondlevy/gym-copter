#!/usr/bin/env python3
'''
Run simple altitude-hold PID controller to test dynamics

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import gym
import numpy as np
import matplotlib.pyplot as plt

DURATION        = 30 # seconds
ALTITUDE_TARGET = 10 # meters

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

        # Compute v setpoint and error
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

def _subplot(t, x, k, label):
    plt.subplot(4,1,k)
    plt.plot(t, x)
    plt.ylabel(label)
 
if __name__ == '__main__':

    # Create and initialize copter environment
    env = gym.make('gym_copter:Copter-v1')
    env.reset()

    # Create PID controller
    pid  = AltitudePidController(ALTITUDE_TARGET, ALT_P, VEL_P, VEL_I, VEL_D)

    # Initialize arrays for plotting
    n = int(DURATION/DT)
    tvals = np.linspace(0, DURATION, n)
    uvals = np.zeros(n)
    zvals = np.zeros(n)
    vvals = np.zeros(n)
    rvals = np.zeros(n)

    # Motors are initially off
    u = 0

    # Loop over time values
    for k,t in np.ndenumerate(tvals):

        # Update the environment with the current motor command
        s, r, _, _ = env.step(u)

        # Extract altitude, vertical velocity from state
        z, v = s

        # Get correction from PID controller
        u = pid.u(z, v, DT)

        # Constrain correction to [0,1] to represent motor value
        u = max(0, min(1, u))

        # Track values
        k = k[0]
        uvals[k] = u
        zvals[k] = z
        vvals[k] = v
        rvals[k] = r

    # Plot results
    _subplot(tvals, rvals, 1, 'Reward')
    _subplot(tvals, zvals, 2, 'Altitude (m)')
    _subplot(tvals, vvals, 3, 'Velocity (m/s)')
    _subplot(tvals, uvals, 4, 'Motors')
    plt.show()

    # Cleanup
    del env
