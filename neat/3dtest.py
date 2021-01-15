#!/usr/bin/env python3
'''
Test script for using NEAT with gym-copter 3D environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import argparse
from argparse import ArgumentDefaultsHelpFormatter
import threading

import gym
from neat_gym import read_file, eval_net
from gym_copter.rendering.threed import ThreeDLanderRenderer

def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    parser.add_argument('--record', default=None, help='If specified, sets the recording dir')
    parser.add_argument('--seed', default=None, type=int, help='Sets Gym, PyTorch and Numpy seeds')
    parser.add_argument('--view', required=False, default=(30,120),
                        help='View elevation, azimuth')
    args = parser.parse_args()

    viewangles = tuple((int(s) for s in args.view.split(',')))

    # Load network and environment name from pickled file
    net, env_name, _, _ = read_file()

    # Make environment from name
    env = gym.make(env_name)

    # Create a three-D renderer
    renderer = ThreeDLanderRenderer(env, viewangles=viewangles)

    # Start the network-evaluation episode on a separate thread
    render, report = True, True
    thread = threading.Thread(target=eval_net, args=(net, env, render, report))
    thread.start()

    # Begin 3D rendering on main thread
    renderer.start() 

if __name__ == '__main__':
    main()
