#!/usr/bin/env python3
'''
3D Copter-Lander heuristic demo support

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from time import sleep
import numpy as np

import gym

from gym_copter.cmdline import make_parser, make_parser_3d, wrap


def _demo_heuristic(env, fun, pidcontrollers, seed=None, csvfilename=None):

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

    while True:

        action = fun(state, pidcontrollers)

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


def demo2d(envname, heuristic, pidcontrollers):

    parser = make_parser()

    args = parser.parse_args()

    env = wrap(args, gym.make(envname))

    _demo_heuristic(env, heuristic, pidcontrollers,
                    seed=args.seed, csvfilename=args.csvfilename)

    env.close()


def demo3d(envname, heuristic, pidcontrollers, renderer):

    env = gym.make(envname)

    parser = make_parser_3d()

    args = parser.parse_args()

    if args.hud:

        env = wrap(args, env)

        env.use_hud()

        _demo_heuristic(env, heuristic, pidcontrollers,
                        args.seed, args.csvfilename)

    else:

        viewangles = tuple((int(s) for s in args.view.split(',')))

        viewer = renderer(env,
                          _demo_heuristic,
                          (heuristic, pidcontrollers,
                           args.seed, args.csvfilename),
                          viewangles=viewangles,
                          outfile='movie.mp4' if args.movie else None)

        viewer.start()
