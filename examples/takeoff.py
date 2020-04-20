#!/usr/bin/env python3
'''
Run simple altitude-hold PID controller to test dynamics

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import gym
import numpy as np
import matplotlib.pyplot as plt

DURATION        = 5  # seconds
ALTITUDE_TARGET = 10 # meters

# PID params
ALT_P = 1.0
VEL_P = 1.0
VEL_I = 0
VEL_D = 0

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
    env = gym.make('gym_copter:Copter-v0')
    env.reset()

    # Create PID controller
    pid  = AltitudePidController(ALTITUDE_TARGET, ALT_P, VEL_P, VEL_I, VEL_D)

    # Initialize arrays for plotting
    tvals = []
    uvals = []
    zvals = []
    vvals = []
    rvals = []

    # Motors are initially off
    u = -1

    # Loop for specified duration
    while True:

        # Get current time from environment
        t = env.time()

        # Stop if time excedes duration
        if t > DURATION: break

        # Update the environment with the current motor command, scaled to [-1,+1] and sent as an array
        s, r, _, _ = env.step([u])

        # Display the environment
        #env.render()

        # Extract altitude, vertical velocity from state
        z, v = s

        # Get correction from PID controller
        u = pid.u(z, v, env.dt)

        # Convert u from [0,1] to [-1,+1]
        u = 2 * u - 1

        # Constrain correction to [-1,+1]
        u = max(-1, min(+1, u))

        # Track values
        tvals.append(t)
        uvals.append(u)
        zvals.append(z)
        vvals.append(v)
        rvals.append(r)

    # Plot results
    _subplot(tvals, rvals, 1, 'Reward')
    _subplot(tvals, zvals, 2, 'Altitude (m)')
    _subplot(tvals, vvals, 3, 'Velocity (m/s)')
    _subplot(tvals, uvals, 4, 'Action')
    plt.ylim([-1.1,+1.1])
    plt.show()

    # Cleanup
    del env
