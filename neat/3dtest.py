#!/usr/bin/env python3
'''
Test script for using NEAT with gym-copter 3D environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import neat
from neat_gym import read_file, eval_net
from gym_copter.rendering.threed import ThreeDLanderRenderer
import threading

if __name__ == '__main__':

    # Load genome and configuration from pickled file
    genome, config = read_file()

    # Make network from genome and configuration
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Create a three-D renderer
    renderer = ThreeDLanderRenderer(config.env)

    # Start the network-evaluation episode on a separate thread
    thread = threading.Thread(target=eval_net, args=(net, config.env, True))
    thread.daemon = True
    thread.start()

    # Begin 3D rendering on main thread
    renderer.start() 
