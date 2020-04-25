#!/usr/bin/env python3
'''
Climb up and leap forward

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import gym
import numpy as np
import matplotlib.pyplot as plt

import gym_copter

DURATION        = 10  # seconds
ALTITUDE_TARGET = 10 # meters

def _subplot(t, x, k, label):
    plt.subplot(2,1,k)
    plt.plot(t, x)
    plt.ylabel(label)


if __name__ == '__main__':

    # Create and initialize copter environment
    env = gym.make('Copter-v1')
    env.reset()

    # Start with motors full-throttle
    u = 1 * np.ones(4)

    # Initialize arrays for plotting
    tvals = []
    uvals = []
    zvals = []
    vvals = []
    rvals = []

    # Loop for specified duration
    while True:

        # Get current time from environment
        t = env.time()

        # Stop if time excedes duration
        if t > DURATION: break

        # Update the environment with the current motor command, scaled to [-1,+1] and sent as an array
        s, r, d, _ = env.step(u)

        # Quit if we're done (crashed)
        if d:
            break

        # Once we reach altitude, switch to forward motion
        z = -s[4]
        if z > ALTITUDE_TARGET:
            u = np.array([0,1,0,1])

        # Track values
        tvals.append(t)
        uvals.append(u)
        zvals.append(z)
        rvals.append(r)

        # Display the environment
        #env.render()

    # Plot results
    _subplot(tvals, rvals, 1, 'Reward')
    _subplot(tvals, zvals, 2, 'Altitude (m)')
    plt.xlabel('Time (sec)')
    plt.show()

    # Cleanup
    del env

