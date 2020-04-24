#!/usr/bin/env python3
'''
Climb up and leap forward

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import gym
import numpy as np

import gym_copter

DURATION        = 5  # seconds
ALTITUDE_TARGET = 10 # meters

# Create and initialize copter environment
env = gym.make('Copter-v1')
env.reset()

# Start with motors full-throttle
u = 1 * np.ones(4)

# Loop for specified duration
while True:

    # Get current time from environment
    t = env.time()

    # Stop if time excedes duration
    if t > DURATION: break

    # Update the environment with the current motor command, scaled to [-1,+1] and sent as an array
    s, r, _, _ = env.step(u)

    # Display the environment
    env.render()

