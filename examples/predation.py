#!/usr/bin/env python3
'''
Visually-guided predation

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import gym
import numpy as np
import threading
import time

ALTITUDE_TARGET = 10 # meters

def update(env):

    # Create and initialize copter environment
    env.reset()

    # Start with motors full-throttle
    u = 1 * np.ones(4)

    # Loop for specified duration
    while True:

        # Update the environment with the current motor command, scaled to [-1,+1] and sent as an array
        s, r, d, _ = env.step(u)

        # Yield to other thread by sleeping for the shortest possible duration
        time.sleep(np.finfo('float').eps)

        # Quit if we're done (crashed)
        if d: break

        # Once we reach altitude, switch to forward motion
        z = -s[4]
        if z > ALTITUDE_TARGET:
            u = np.array([0,1,0,1])

    # Cleanup
    del env

if __name__ == '__main__':

    # Create environment
    env = gym.make('gym_copter:CopterPredator-v0')

    plotter = env.tpvplotter()

    # Run simulation on its own thread
    thread = threading.Thread(target=update, args=(env,))
    thread.daemon = True
    thread.start()

    # Begin 3D rendering on main thread
    plotter.start()

