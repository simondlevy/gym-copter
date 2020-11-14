#!/usr/bin/env python3
'''
Test script for NEAT CopterLanderV2

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import visualize
import pickle
from sys import argv

from common import eval_genome

if __name__ == '__main__':

    if len(argv) < 2:
        print('Usage:   %s FILENAME' % argv[0])
        exit(1)

    genome, config = pickle.load(open(argv[1], 'rb'))

    config.reps = 1

    print('%6.6f' % eval_genome(genome, config, True))

    node_names = {-1:'x', -2:'dx', -3:'y', -4:'dy', -5:'phi', -6:'dphi', 0:'mr', 1:'ml'}
    visualize.draw_net(config, genome, True, node_names = node_names)
