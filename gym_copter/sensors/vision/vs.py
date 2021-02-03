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

    def get_image(self, pose):

        # Extract pose elements
        x, y, z, _phi, _theta, _psi = pose

        image = np.zeros((self.resolution, )*2)

        cx = self.locate(z, x)
        cy = self.locate(z, y)
        cr = self.scale(z, self.object_size)

        # Add a circle with radius and line thickness proportional to fraction
        # of object in current field of view.
        cv2.circle(image, (cx, cy), cr, 1, thickness=self.scale(z, 0.1))

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

    parser.add_argument('--file',  default='chevron.png', help='Input file')
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

    vs = VisionSensor()

    image = vs.get_image((0, 0, 10, 0, 0, 0))

    while True:

        if not VisionSensor.display_image(image, 'Image'):
            break

if __name__ == '__main__':

    main()
