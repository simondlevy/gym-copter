#!/usr/bin/env python3
'''
Heuristic demo for 2D Copter hovering

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from time import sleep
import gym
import numpy as np


def _constrain(val, lim):
    return -lim if val < -lim else (+lim if val > +lim else val)


def main():

    k_start = 10
    k_p = 0.2
    k_i = 3
    k_tgt = 5
    k_windup = 0.2

    # Error integral
    ei = 0

    # Start CSV file
    filename = ('kp=%2.2f_Ki=%2.2f_k_tgt=%2.2f_k_windup=%2.2f.csv' %
                (k_p, k_i, k_tgt, k_windup))
    csvfile = open(filename, 'w')
    csvfile.write('z,dz,e,ei,u\n')

    env = gym.make('gym_copter:Hover1D-v0')

    env.set_altitude(3)

    total_reward = 0
    steps = 0
    state = env.reset()

    while steps < 500:

        z, dz = state

        # Negate for NED => ENU
        z, dz = -z, -dz

        # Compute error as scaled target minus actual
        e = (k_tgt - z) - dz

        # Compute I term
        ei += e

        # Avoid integral windup
        ei = _constrain(ei, k_windup)

        # Compute demand u
        u = e * k_p + ei * k_i

        # Constrain u to interval [0,1].  This is done automatically
        # by our gym environment, but we do it here to avoid writing
        # out-of-bound values to the CSV file.
        u = np.clip(u, 0, 1)

        # Write current values to CSV file
        csvfile.write('%3.3f,%3.3f,%3.3f,%3.3f,%3.3f\n' % (z, dz, e, ei, u))

        state, reward, done, _ = env.step((u,))

        total_reward += reward

        env.render()

        sleep(1./env.FRAMES_PER_SECOND)

        steps += 1

        if (steps % 20 == 0) or done:
            print('steps =  %04d    total_reward = %+0.2f' %
                  (steps, total_reward))

        if done:
            break

    env.close()


main()
