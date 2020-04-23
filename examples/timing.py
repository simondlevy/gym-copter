#!/usr/bin/env python3
'''
Time the update rate

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import gym
from time import time

NITER = 10000

if __name__ == '__main__':

    # Create and initialize the simplest copter environment (on/off motors)
    env = gym.make('gym_copter:Copter-v0')
    env.reset()

    start = time()

    # Loop a bunch of times
    for k in range(NITER):

        # Run full-throttle
        env.step([1])

    del env

    print('%d fps' % (int(NITER/(time()-start))))
