#!/usr/bin/env python3
'''
Script for plotting results 3D gym-copter run

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import argparse
from argparse import ArgumentDefaultsHelpFormatter
import numpy as np
import matplotlib.pyplot as plt

from gym_copter.envs.lander import _Lander

def main():

    parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('csvfile', metavar='CSVFILE', help='input .csv file')

    parser.add_argument('--raw', action='store_true',
                        help='File has no header or timestamps')

    args = parser.parse_args()

    try:
        data = np.genfromtxt(args.csvfile, delimiter=',',
                             skip_header=(0 if args.raw else 1))
    except Exception:
        print('Unable to open file %s' % args.csvfile)
        exit(1)

    zcol = 5
    t = data[:,0]

    if args.raw:
        n = data.shape[0]
        dur = n / _Lander.FRAMES_PER_SECOND
        t = np.linspace(0, dur, n) if args.raw else data[:, 0]
        zcol = 4

    z = data[:, zcol]
    dz = data[:, (zcol+1)]

    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    axs[0].plot(t, -z)  # adjust for NED
    axs[0].set_ylabel('Z (m)')

    fig.suptitle(args.csvfile, fontsize=16)

    axs[1].plot(t, -dz)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('dZ/dt (m/s)')

    plt.show()


main()
