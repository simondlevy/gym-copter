#!/usr/bin/env python3
'''
Run simple takeoff maneuver to test gym-copter

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import gym
import numpy as np

if __name__ == '__main__':

    # Create and initialize the simplest copter environment (on/off motors)
    env = gym.make('gym_copter:Copter-v0')
    env.reset()

    # Loop until user hits the stop button
    while True:

        # Draw the current environment
        if env.render() is None: break

        # Turn the motors on
        env.step(1)

    del env
