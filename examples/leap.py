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

if __name__ == '__main__':

    # Create and initialize copter environment
    env = gym.make('Copter-v1')
    env.reset()

    # Start with motors full-throttle
    u = 1 * np.ones(4)

    # Loop for specified duration
    while True:

        # Get current time from environment
        t = env.time()

        # Update the environment with the current motor command, scaled to [-1,+1] and sent as an array
        s, r, d, _ = env.step(u)

        # Quit if we're done (crashed)
        if d: break

        # Once we reach altitude, switch to forward motion
        z = -s[4]
        if z > ALTITUDE_TARGET:
            u = np.array([0,1,0,1])

        # Display the environment
        env.render()

    # Cleanup
    del env

