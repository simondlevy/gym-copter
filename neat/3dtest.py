#!/usr/bin/env python3
'''
Test script for using NEAT with gym-copter 3D environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import gym
from neat_gym import read_file, eval_net
from gym_copter.rendering.threed import ThreeDLanderRenderer
import threading

if __name__ == '__main__':

    # Load network and environment name from pickled file
    net, env_name, _, _ = read_file()

    # Make environment from name
    env = gym.make(env_name)

    # Create a three-D renderer
    renderer = ThreeDLanderRenderer(env)

    # Start the network-evaluation episode on a separate thread
    thread = threading.Thread(target=eval_net, args=(net, env))
    thread.daemon = True
    thread.start()

    # Begin 3D rendering on main thread
    renderer.start() 
