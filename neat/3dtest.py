#!/usr/bin/env python3
'''
Test script for using NEAT with gym-copter 3D environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import threading
import pickle

import gym
from gym_copter.rendering.threed import ThreeDLanderRenderer
from gym_copter.rendering.threed import make_parser, parse

from neat_gym import eval_net


def main():

    # Make a command-line parser with --view enabled
    parser = make_parser()

    parser.add_argument('filename', metavar='FILENAME', help='.dat input file')
    parser.add_argument('--movie', default=None,
                        help='If specified, sets the output movie file name')
    args, viewangles = parse(parser)

    # Load net and environment name from pickled file
    net, env_name = pickle.load(open(args.filename, 'rb'))

    # Make environment from name
    env = gym.make(env_name)

    # Start the network-evaluation episode on a separate thread
    render, report = True, True
    thread = threading.Thread(target=eval_net, args=(net, env, render, report))
    thread.start()

    # Begin 3D rendering on main thread
    renderer = ThreeDLanderRenderer(env, viewangles=viewangles,
                                    outfile=args.movie)
    renderer.start()


if __name__ == '__main__':
    main()
