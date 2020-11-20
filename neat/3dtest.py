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
import numpy as np
import time

if __name__ == '__main__':

    # Load genome and configuration from pickled file
    genome, config = read_file()

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    renderer = ThreeDLanderRenderer(config.env)

    thread = threading.Thread(target=eval_net, args=(net, config.env))
    thread.daemon = True
    thread.start()

    # Begin 3D rendering on main thread
    renderer.start() 
