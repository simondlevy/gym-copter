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

    parser.add_argument('--title', required=False, default=None,
                        help='Figure title (defaults to filename)')

    parser.add_argument('--time', type=float, default=8,
                        help='Time axis limit')

    parser.add_argument('--dzlim', type=float, default=15,
                        help='Axis limit for dZ/dt')

    args = parser.parse_args()

    data = None

    try:
        data = np.genfromtxt(args.csvfile, delimiter=',')

    except Exception as e:
        print('Unable to open file %s: %s' % (args.csvfile + str(e)))
        exit(1)

    if data.shape[1] == 15:
        t = data[1:, 0]
        data = data[1:, 1:]

    else:
        n = data.shape[0]
        dur = n / _Lander.FRAMES_PER_SECOND
        t = np.linspace(0, dur, n)

    z = data[:, 8]
    dz = data[:, 9]

    fig, axs = plt.subplots(3, 1, constrained_layout=True)

    axs[0].plot(t, -z)  # adjust for NED
    axs[0].set_ylabel('Z (m)')

    fig.suptitle(args.csvfile if args.title is None else args.title,
                 fontsize=16)

    axs[1].plot(t, -dz)
    axs[1].set_ylim((0, -args.dzlim))
    axs[1].set_ylabel('dZ/dt (m/s)')

    motors = data[:, 0:4]
    for k in range(4):
        axs[2].plot(t, motors[:, k])
    axs[2].set_ylabel('Motors')
    axs[2].set_ylim((0, 1))

    axs[2].set_xlabel('Time (s)')

    for ax in axs:
        ax.set_xlim((0, args.time))

    plt.show()


main()
