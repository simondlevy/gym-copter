#!/usr/bin/env python3
'''
Test script for using NEAT with gym-copter 3D environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import threading
import pickle

from gym_copter.cmdline import make_parser_3d, parse_view_angles

import gymnasium as gym
from gym_copter.rendering.threed import ThreeDLanderRenderer

from neat_gym import eval_net


def eval_with_movie(env, net, seed):
    eval_net(net, env, render=True, seed=seed)

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

    movie_name = None

    if args.movie:
        print('Running episode ...')
        movie_name = 'movie.mp4'

    # Begin 3D rendering on main thread
    # render, report = True, True
    renderer = ThreeDLanderRenderer(env,
                                    eval_with_movie,
                                    (net, args.seed),
                                    viewangles=viewangles,
                                    outfile=movie_name)
    renderer.start()


if __name__ == '__main__':
    main()
