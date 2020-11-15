#!/usr/bin/env python3
'''
Test script for NEAT CopterLanderV2

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import visualize
import pickle
import argparse

from common import eval_genome

if __name__ == '__main__':

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    args = parser.parse_args()

    # Load genome and configuration from pickled file
    genome, config = pickle.load(open(args.filename, 'rb'))

    node_names = {-1:'x', -2:'dx', -3:'y', -4:'dy', -5:'phi', -6:'dphi', 0:'mr', 1:'ml'}
    visualize.draw_net(config, genome, True, node_names = node_names)
