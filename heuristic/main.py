#!/usr/bin/env python3
'''
3D Copter-Lander heuristic demo support

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import gym
from gym import wrappers

from parsing import make_parser
from heuristic import demo_heuristic


def demo2d(envname, heuristic, pidcontrollers):
    
    parser = make_parser()

    args = parser.parse_args()

    env = gym.make(envname)

    env = wrappers.Monitor(env, 'movie/', force=True)

    demo_heuristic(env, heuristic, pidcontrollers,
                   seed=args.seed, csvfilename=args.csvfilename)

    env.close()


def demo3d(env, heuristic, pidcontrollers, renderer):

    parser = make_parser()

    group = parser.add_mutually_exclusive_group()

    group.add_argument('--hud', action='store_true',
                       help='Use heads-up display')

    group.add_argument('--view', required=False, default='30,120',
                       help='Elevation, azimuth for view perspective')

    args = parser.parse_args()

    if args.hud:

        env.use_hud()

        demo_heuristic(env, heuristic, pidcontrollers,
                       args.seed, args.csvfilename)

    else:

        viewangles = tuple((int(s) for s in args.view.split(',')))

        viewer = renderer(env,
                          demo_heuristic,
                          (heuristic, pidcontrollers,
                           args.seed, args.csvfilename),
                          viewangles=viewangles)

        viewer.start()
