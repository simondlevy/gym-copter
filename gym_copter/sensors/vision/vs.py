#!/usr/bin/env python3
'''
Vision Sensor simulation

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import argparse
from argparse import ArgumentDefaultsHelpFormatter
import numpy as np
import cv2


class VisionSensor(object):

    def __init__(self, object_size=1, resolution=128, field_of_view=60):
        '''
        @param object_size meters
        @param resolution pixels
        @param field_of_view degrees
        '''

        self.object_size = object_size
        self.resolution = resolution

        self.tana = np.tan(np.radians(field_of_view/2))

    def get_image(self, x, y, z, phi, theta, psi):

        image = np.zeros((self.resolution, )*2)

        # Compute image center in pixels
        cx = self.locate(z, x)
        cy = self.locate(z, y)

        # A pentagon centered around the origin with unit width and height
        shape = np.array([[-.5, -0.0714],
                          [0, -0.5],
                          [+.5, -0.0714],
                          [+.25, +.5],
                          [-.25, +.5]])

        # Scale up the pentagon and center it in the image
        shape *= self.scale(z, self.object_size)
        shape[:, 0] += cx
        shape[:, 1] += cy

        # Draw the shapegon as a filled polygon
        cv2.fillPoly(image, [shape.astype('int32')], 255)

        return image

    def locate(self, z, coord):

        return self.scale(z, coord) + self.resolution//2

    def scale(self, z, val):

        return int(val * self.resolution / (2 * z * self.tana))

    @staticmethod
    def display_image(image, name, scaleup=4):
        '''
        Scale up and display the image
        '''
        image = cv2.resize(image, ((128*scaleup, )*2))
        cv2.imshow(name, image)
        return cv2.waitKey(10) != 27  # ESC


# End of VisionSensor classes -------------------------------------------------


def main():

    parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--x',  type=float, default=0, help='X coordinate (m)')
    parser.add_argument('--y',  type=float, default=0, help='Y coordinate (m)')
    parser.add_argument('--z',  type=float, default=5, help='Z coordinate (m)')
    parser.add_argument('--phi',  type=float, default=0,
                        help='Roll angle (deg)')
    parser.add_argument('--theta',  type=float, default=0,
                        help='Pitch angle (deg)')
    parser.add_argument('--psi',  type=float, default=0,
                        help='Yaw angle (deg)')
    parser.add_argument('--fov',  type=float, default=30,
                        help='Field of view (deg)')
    parser.add_argument('--size',  type=float, default=1,
                        help='Object size (m)')

    args = parser.parse_args()

    vs = VisionSensor(args.size)

    image = vs.get_image(args.x, args.y, args.z,
                         args.phi, args.theta, args.psi)

    while True:

        if not VisionSensor.display_image(image, 'Image'):
            break


if __name__ == '__main__':

    main()
