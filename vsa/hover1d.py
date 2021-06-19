#!/usr/bin/env python3
'''
Heuristic demo for 2D Copter hovering

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from time import sleep
import numpy as np

import gym

from gym_copter.cmdline import make_parser


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

    def getDemand(self, z, dz):

        # Negate for NED => ENU
        z, dz = -z, -dz

        # Compute error as scaled target minus actual
        e = (self.k_tgt - z) - dz

        # Compute I term
        self.ei += e

        # avoid integral windup
        self.ei = _constrain(self.ei, self.k_windup)

        return e * self.k_p + self.ei * self.k_i


def heuristic(state, pidcontrollers):

    z, dz = state

    alt_pid = pidcontrollers[0]

    return (alt_pid.getDemand(z, dz),)


def _demo_heuristic(env, fun, pidcontrollers,
                    seed=None, csvfilename=None, nopid=False):

    env.seed(seed)
    np.random.seed(seed)

    total_reward = 0
    steps = 0
    state = env.reset()

    dt = 1. / env.FRAMES_PER_SECOND

    actsize = env.action_space.shape[0]

    csvfile = None
    if csvfilename is not None:
        csvfile = open(csvfilename, 'w')
        csvfile.write('t,' + ','.join([('m%d' % k)
                                      for k in range(1, actsize+1)]))
        csvfile.write(',' + ','.join(env.STATE_NAMES) + '\n')

    while steps < 500:

        action = np.zeros(actsize) if nopid else fun(state, pidcontrollers)

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if csvfile is not None:

            csvfile.write('%f' % (dt * steps))

            csvfile.write((',%f' * actsize) % tuple(action))

            csvfile.write(((',%f' * len(state)) + '\n') % tuple(state))

        env.render()

        sleep(1./env.FRAMES_PER_SECOND)

        steps += 1

        if (steps % 20 == 0) or done:
            print('steps =  %04d    total_reward = %+0.2f' %
                  (steps, total_reward))

        if done:
            break

    env.close()

    if csvfile is not None:
        csvfile.close()


def demo(envname, heuristic, pidcontrollers):

    parser = make_parser()

    args = parser.parse_args()

    env = gym.make(envname)

    _demo_heuristic(env, heuristic, pidcontrollers,
                    seed=args.seed, csvfilename=args.csvfilename,
                    nopid=args.nopid)

    env.close()


demo('gym_copter:Hover1D-v0', heuristic, (AltitudeHoldPidController(),))
