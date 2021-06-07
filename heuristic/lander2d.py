#!/usr/bin/env python3
'''
2D Copter lander

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from time import time, sleep

import numpy as np
import gym
from gym import wrappers

from parsing import make_parser
from pidcontrollers import AngularVelocityPidController
from pidcontrollers import PositionHoldPidController
from pidcontrollers import AltitudeHoldPidController


def heuristic(state, rate_pid, poshold_pid, descent_pid):

    y, dy, z, dz, phi, dphi = state

    phi_todo = 0

    rate_todo = rate_pid.getDemand(dphi)
    pos_todo = poshold_pid.getDemand(y, dy)

    phi_todo = rate_todo + pos_todo

    hover_todo = descent_pid.getDemand(z, dz)

    return hover_todo-phi_todo, hover_todo+phi_todo


def demo_heuristic(env, seed=None, csvfilename=None):

    env.seed(seed)
    np.random.seed(seed)

    rate_pid = AngularVelocityPidController()
    pos_pid = PositionHoldPidController()
    alt_pid = AltitudeHoldPidController()

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

    while True:

        action = heuristic(state, rate_pid, pos_pid, alt_pid)
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if csvfile is not None:

            csvfile.write('%f' % (dt * steps))

            csvfile.write((',%f' * actsize) % tuple(action))

            csvfile.write(((',%f' * len(state)) + '\n') % tuple(state))

        steps += 1

        if (steps % 20 == 0) or done:
            print('time = %3.2f   steps =  %04d    total_reward = %+0.2f' %
                  (time()-env.start, steps, total_reward))

        if done:
            break

    sleep(1)
    env.close()
    if csvfile is not None:
        csvfile.close()
    return total_reward


def main():

    parser = make_parser()

    args = parser.parse_args()

    env = gym.make('gym_copter:Lander2D-v0')

    env = wrappers.Monitor(env, 'movie/', force=True)

    demo_heuristic(env, seed=args.seed, csvfilename=args.csvfilename)

    env.close()


if __name__ == '__main__':
    main()
