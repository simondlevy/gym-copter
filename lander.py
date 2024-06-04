#!/usr/bin/env python3
'''
Heuristic demo of a landing task

Copyright (C) 2024 Simon D. Levy

MIT License
'''

import argparse
from argparse import ArgumentDefaultsHelpFormatter

from time import sleep

import numpy as np

import gymnasium as gym

from gym_copter.rendering import ThreeDLanderRenderer

MOTORVAL = 1.625e-2


# Threaded
def heuristic(env, csvfilename=None, random=False):

    total_reward = 0
    steps = 0
    state, _ = env.reset()

    dt = 1. / env.unwrapped.FRAMES_PER_SECOND

    csvfile = None
    if csvfilename is not None:
        csvfile = open(csvfilename, 'w')
        csvfile.write('t,' + ','.join([('m%d' % k)
                                      for k in range(1, 5)]))
        csvfile.write(',' + ','.join(env.STATE_NAMES) + '\n')

    while True:

        action = MOTORVAL  * (np.random.randn(4) if random else np.ones(4))

        state, reward, done, _, _ = env.step(action)

        total_reward += reward

        if csvfile is not None:

            csvfile.write('%f' % (dt * steps))

            csvfile.write((',%f' * 4) % tuple(action))

            csvfile.write(((',%f' * len(state)) + '\n') % tuple(state))

        env.render()

        sleep(1./env.unwrapped.FRAMES_PER_SECOND)

        steps += 1

        print('steps =  %04d    total_reward = %+0.2f' % (steps, total_reward))

        if done:
            break

    env.close()

    if csvfile is not None:
        csvfile.close()


def parse_view_angles(args):

    return tuple((int(s) for s in args.view.split(',')))


def main():

    env = gym.make('gym_copter:Lander-v0')

    parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--save', dest='csvfilename',
                        help='Save trajectory in CSV file')

    parser.add_argument('--movie', action='store_true',
                        help='Save movie in an MP4 file')

    group = parser.add_mutually_exclusive_group()

    group.add_argument('--view', required=False, default='30,120',
                       help='Elevation, azimuth for view perspective')

    parser.add_argument('--random', action='store_true',
                        help='Use random motor values for comparison')

    args = parser.parse_args()

    viewer = ThreeDLanderRenderer(env,
                                  heuristic,  # threadfun
                                  (args.csvfilename, args.random), # threadargs
                                  viewangles=parse_view_angles(args),
                                  outfile='movie.mp4' if args.movie else None)

    viewer.start()


if __name__ == '__main__':

    main()
