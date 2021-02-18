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

    DZ_AXLIM = 15

    parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('csvfile', metavar='CSVFILE', help='input .csv file')

    parser.add_argument('--raw', action='store_true',
                        help='File has no header or timestamps')

    parser.add_argument('--title', required=False, default=None,
                        help='Figure title (defaults to filename)')

    args = parser.parse_args()

    try:
        data = np.genfromtxt(args.csvfile, delimiter=',',
                             skip_header=(0 if args.raw else 1))
    except Exception:
        print('Unable to open file %s' % args.csvfile)
        exit(1)

    col = 1
    t = data[:, 0]

    if args.raw:
        n = data.shape[0]
        dur = n / _Lander.FRAMES_PER_SECOND
        t = np.linspace(0, dur, n) if args.raw else data[:, 0]
        col = 0

    z = data[:, col+8]
    dz = data[:, (col+9)]

    fig, axs = plt.subplots(3, 1, constrained_layout=True)

    axs[0].plot(t, -z)  # adjust for NED
    axs[0].set_ylabel('Z (m)')

    fig.suptitle(args.csvfile if args.title is None else args.title,
                 fontsize=16)

    axs[1].plot(t, -dz)
    axs[1].set_ylim((0, -DZ_AXLIM))
    axs[1].set_ylabel('dZ/dt (m/s)')

    motors = data[:, col:(col+4)]
    for k in range(4):
        axs[2].plot(t, motors[:, k])
    axs[2].set_ylabel('Motors')

    axs[2].set_xlabel('Time (s)')

    plt.show()


main()
