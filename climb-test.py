#!/usr/bin/env python3
'''
Copyright (C) 2019 Simon D. Levy

MIT License
'''

import gym
import numpy as np
from time import time

if __name__ == '__main__':

    # Create and initialize copter environment
    env = gym.make('gym_copter:Copter-v1')
    env.reset()

    prev = time()

    # Loop until user hits the stop button
    while True:

        # Draw the current environment
        if env.render() is None: break

        dt = time() - prev

        # Update the environment with the current motor commands
        state, _, _, _ = env.step(np.ones(4), dt)

        print('dt=%3.3f  Alt=%3.2fm' % (dt, -state[5]))

        prev = time()

    del env
