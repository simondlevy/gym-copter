#!/usr/bin/env python3
'''
Run simple constant-climb-rate PID controller to test dynamics

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import gym
import numpy as np
import matplotlib.pyplot as plt
from sys import stdout

DURATION = 10   # seconds
DT       = .001 # seconds
TARGET   = 1  # meters per second

# PID params
P = 0.1
I = 0.1
D = 0.1

class AltitudePidController(object):

    def __init__(self, target, P, I, D, windupMax=10):

        # In a real PID controller, this would be a set-point
        self.target = target

        # Constants
        self.P = P
        self.I = I
        self.D = D
        self.windupMax = windupMax

        # Values modified in-flight
        self.lastError      = 0
        self.integralError  = 0

    def u(self, vel, dt):

        # Compute dzdt setpoint and error
        error = self.target - vel

        # Update error integral and error derivative
        self.integralError +=  error * dt
        self.integralError = AltitudePidController._constrainAbs(self.integralError + error * dt, self.windupMax)
        deltaError = (error - self.lastError) / dt if abs(self.lastError) > 0 else 0
        self.lastError = error

        # Compute control u
        return self.P * error + self.D * deltaError + self.I * self.integralError

    def _constrainAbs(x, lim):

        return -lim if x < -lim else (+lim if x > +lim else x)

if __name__ == '__main__':

    # Create and initialize copter environment
    env = gym.make('gym_copter:Copter-v2')
    env.reset()

    # Create PID controller
    pid  = AltitudePidController(TARGET, P, I, D)

    # Initialize arrays for plotting
    n = int(DURATION/DT)
    tvals = np.linspace(0, DURATION, n)
    uvals = np.zeros(n)
    vvals = np.zeros(n)

    # Motors are initially off
    u = 0

    # Loop until user hits the stop button
    for k,t in np.ndenumerate(tvals):

        # Update the environment with the current motor commands
        state, _, _, _ = env.step(u*np.ones(4))

        # Extract velocity from state (negate to accommodate NED)
        v = -state[5]

        # Get correction from PID controller
        u = pid.u(v, DT)

        # Constrain correction to [0,1] to represent motor value
        u = max(0, min(1, u))

        # Track values
        k = k[0]
        uvals[k] = u
        vvals[k] = v

    # Plot results
    plt.subplot(2,1,1)
    plt.plot(tvals, vvals)
    plt.ylabel('Velocity (m/s)')
    plt.subplot(2,1,2)
    plt.plot(tvals, uvals)
    plt.ylabel('Motors')
    plt.xlabel('Time (s)')
    plt.show()

    # Cleanup
    del env
