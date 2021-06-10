#!/usr/bin/env python3
'''
Command-line parsing utitities for gym-copter

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import argparse
from argparse import ArgumentDefaultsHelpFormatter

import gym
from gym import wrappers


def make_parser(envname):

    parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', type=int, required=False, default=None,
                        help='Random seed for reproducibility')

    parser.add_argument('--nopid', action='store_true',
                        help='Turn off lateral PID control')

    parser.add_argument('--save', dest='csvfilename',
                        help='Save trajectory in CSV file')

    parser.add_argument('--movie', action='store_true',
                        help='Save movie in an MP4 file')

    return parser, gym.make(envname)


def wrap(args, env):

    return (env if args.movie is None
            else wrappers.Monitor(env, 'movie/', force=True))


def make_parser_3d(envname):

    parser, env = make_parser(envname)

    group = parser.add_mutually_exclusive_group()

    group.add_argument('--hud', action='store_true',
                       help='Use heads-up display')

    group.add_argument('--view', required=False, default='30,120',
                       help='Elevation, azimuth for view perspective')

    return parser, env
