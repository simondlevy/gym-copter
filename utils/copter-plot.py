#!/usr/bin/env python3
'''
Script for plotting results 3D gym-copter run

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', metavar='CSVFILE', help='input .csv file')
    args = parser.parse_args()

    try:
        data = np.genfromtxt(args.csvfile, delimiter=',', skip_header=1)
    except Exception:
        print('Unable to open file %s' % args.csvfile)
        exit(1)

    print(data)

    plt.title(args.csvfile)
    plt.show()


main()
