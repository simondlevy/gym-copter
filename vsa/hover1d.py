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


class AltitudeHoldPidController:

    def __init__(self, k_p=0.2, k_i=3, k_tgt=5, k_windup=0.2):

        self.k_p = k_p
        self.k_i = k_i

        self.k_tgt = k_tgt

        # Prevents integral windup
        self.k_windup = k_windup

        # Error integral
        self.ei = 0

        # Start CSV file
        filename = ('kp=%2.2f_Ki=%2.2f_k_tgt=%2.2f_k_windup=%2.2f.csv' %
                    (k_p, k_i, k_tgt, k_windup))
        self.csvfile = open(filename, 'w')
        self.csvfile.write('z,dz,e,ei,u\n')

    def getDemand(self, z, dz):

        # Negate for NED => ENU
        z, dz = -z, -dz

        # Compute error as scaled target minus actual
        e = (self.k_tgt - z) - dz

        # Compute I term
        self.ei += e

        # Avoid integral windup
        self.ei = _constrain(self.ei, self.k_windup)

        # Compute demand u
        u = e * self.k_p + self.ei * self.k_i

        # Constrain u to interval [0,1].  This is done automatically
        # by our gym environment, but we do it here to avoid writing
        # out-of-bound values to the CSV file.
        u = np.clip(u, 0, 1)

        # Write current values to CSV file
        self.csvfile.write('%3.3f,%3.3f,%3.3f,%3.3f,%3.3f\n' %
                           (z, dz, e, self.ei, u))

        return u


def main():

    env = gym.make('gym_copter:Hover1D-v0')

    env.set_altitude(3)

    total_reward = 0
    steps = 0
    state = env.reset()

    alt_pid = AltitudeHoldPidController()

    while steps < 500:

        z, dz = state

        action = alt_pid.getDemand(z, dz)

        state, reward, done, _ = env.step((action,))

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
