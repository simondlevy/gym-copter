#!/usr/bin/env python3
'''
Run simple constant-climb-rate PID controller to test dynamics

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import gym
import numpy as np
from time import time
from sys import stdout

# Target (meters per second)
TARGET = 1

# PID params
P = 1.0
I = 0
D = 0

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

    # initial conditions
    z = 0
    zprev = 0
    u = np.zeros(4)

    # Create PID controller
    pid  = AltitudePidController(TARGET, P, I, D)

    # Start timing
    prev = time()

    # Loop until user hits the stop button
    while True:

        # Draw the current environment
        if env.render() is None: break

        # Update timer
        curr = time()
        dt = curr - prev
        prev = curr

        # Update the environment with the current motor commands
        state, _, _, _ = env.step(u*np.ones(4))

        # Extract velocity from state (negate to accommodate NED)
        vel = -state[5]

        # Get correction from PID controller
        u = pid.u(vel, dt)

        # Constrain correction to [0,1] to represent motor value
        u = max(0, min(1, u))

        print('%+3.3f %1.1f' % (vel, u))
        stdout.flush()

    del env
