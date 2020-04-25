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

ALTITUDE_TARGET = 10 # meters

def _subplot(t, x, k, label):
    plt.subplot(3,1,k)
    plt.plot(t, x)
    plt.ylabel(label)


if __name__ == '__main__':

    # Create and initialize copter environment
    env = gym.make('Copter-v1')
    env.reset()

    # Start with motors full-throttle
    u = 1 * np.ones(4)

    # Initialize arrays for plotting
    u1vals = []
    u2vals = []
    u3vals = []
    u4vals = []
    tvals = []
    uvals = []
    zvals = []
    vvals = []
    rvals = []

    # Loop for specified duration
    while True:

        # Get current time from environment
        t = env.time()

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
        u1vals.append(u[0])
        u2vals.append(u[1])
        u3vals.append(u[2])
        u4vals.append(u[3])
        tvals.append(t)
        uvals.append(u)
        zvals.append(z)
        rvals.append(r)

        # Display the environment
        env.render()

    '''
    # Plot results
    plt.figure()
    _subplot(tvals, u1vals, 1, 'Action')
    _subplot(tvals, u2vals, 1, 'Action')
    _subplot(tvals, u3vals, 1, 'Action')
    _subplot(tvals, u4vals, 1, 'Action')
    plt.legend(['m1','m2','m3','m4'])
    _subplot(tvals, rvals, 2, 'Reward')
    _subplot(tvals, zvals, 3, 'Altitude (m)')
    plt.xlabel('Time (sec)')
    plt.show()
    '''

    # Cleanup
    del env

