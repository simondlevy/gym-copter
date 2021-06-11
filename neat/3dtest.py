#!/usr/bin/env python3
'''
Test script for using NEAT with gym-copter 3D environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import threading
import pickle

from gym_copter.cmdline import make_parser_3d, parse_view_angles

import gym
from gym_copter.rendering.threed import ThreeDLanderRenderer

from neat_gym import eval_net


def eval_with_movie(net, env, render, report, seed, movie, csvfilename):
    eval_net(net, env,
             seed=seed, render=render, report=report, csvfilename=csvfilename)
    if movie is not None:
        print('Saving %s ...' % movie)


def main():

    # Make a command-line parser
    parser = make_parser_3d()
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    args = parser.parse_args()
    viewangles = parse_view_angles(args)

     # Load net and environment name from pickled file
    net, env_name = pickle.load(open(args.filename, 'rb'))

    # Make environment from name
    env = gym.make(env_name)

    if args.movie is not None:
        print('Running episode ...')

    exit(0)

    # Start the network-evaluation episode on a separate thread
    render, report = True, True
    thread = threading.Thread(target=eval_with_movie,
                              args=(net,
                                    env,
                                    render,
                                    report,
                                    args.seed,
                                    args.movie,
                                    args.csvfilename))
    thread.start()

    # Begin 3D rendering on main thread
    renderer = ThreeDLanderRenderer(env, viewangles=viewangles,
                                    outfile=args.movie)
    renderer.start()


if __name__ == '__main__':
    main()
