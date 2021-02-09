#!/usr/bin/env python3
'''
Dynamic Vision Sensor simulation

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import argparse
from argparse import ArgumentDefaultsHelpFormatter
import numpy as np
import cv2

from gym_copter.sensors.vision.vs import VisionSensor, _make_parser


class DVS(VisionSensor):

    def __init__(self, objsize=1, res=128, fov=60):
        '''
        @param size size meters
        @param res resolution in (pixels)
        @param fov field of view (degrees)
        '''

        VisionSensor.__init__(self, objsize, res, winname='DVS')

        self.image_prev = None

    def getImage(self, x, y, z, phi, theta, psi):
        '''
        @param x, y, z position (meters)
        @param phi, theta, psi Euler angles (degrees)
        '''

        image_curr = VisionSensor.getImage(self, x, y, z, phi, theta, psi)

        # First time around, no eventsimage.  Subsequent times, do a first
        # difference to get the events.
        image_diff = (self.image_prev - image_curr
                      if self.image_prev is not None
                      else np.zeros((self.res, self.res)))

        # Quantize image to -1, 0, +1
        image_diff[image_diff > 0] = +1
        image_diff[image_diff < 0] = -1

        # Track previous event image for first difference
        self.image_prev = image_curr

        return image_diff 

    def _process_image(self, image):

        # Make a color image with -1 red and +1 green
        cimage = np.zeros((image.shape[0], image.shape[1], 3)).astype('uint8')
        self._set_color(image, cimage, -1, 2)
        self._set_color(image, cimage, +1, 1)

        return cimage

    def _set_color(self, src, dst, val, chan):
        r, c = np.where(src == val)
        dst[r,c,chan] = 255
 

# End of DVS class -------------------------------------------------


def main():

    XRANGE = 4
    SPEED = .02

    parser = _make_parser()
    args = parser.parse_args()

     # Arbitrary object size (1m)
    dvs = DVS(args.objsize, args.res, args.fov)

    # Arbitrary stating pose
    x, y, z, phi, theta, psi = -XRANGE, 0, 10, 0, 0, 0

    dx = +1

    while True:

        image = dvs.getImage(x, y, z, phi, theta, psi)

        if not dvs.display_image(image, 'Events'):
            break

        # Move pose across field of view
        x += dx * SPEED

        if x <= -XRANGE:
            dx = +1

        if x >= XRANGE:
            dx = -1

if __name__ == '__main__':

    main()
