'''
Copyright (C) 2021 Simon D. Levy

MIT License
'''

import argparse
from argparse import ArgumentDefaultsHelpFormatter


def make_parser():

    parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', type=int, required=False, default=None,
                        help='Random seed for reproducibility')

    parser.add_argument('--nopid', action='store_true',
                        help='Turn off lateral PID control')

    parser.add_argument('--save', dest='csvfilename',
                        help='Save trajectory in CSV file')

    return parser

def make_3d_parser():

    parser = make_parser()

    group = parser.add_mutually_exclusive_group()

    group.add_argument('--hud', action='store_true',
                       help='Use heads-up display')

    group.add_argument('--view', required=False, default='30,120',
                       help='Elevation, azimuth for view perspective')

    return parser
