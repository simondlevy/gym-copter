#!/usr/bin/env python3
'''
Vision Sensor simulation

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np
import cv2


class VisionSensor(object):

    def __init__(self, object_size, resolution=128, field_of_view=60):
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
        x, y, z, _phi, _theta = pose

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

    vs = VisionSensor(1)

    # Arbitrary stating position
    pose = [0, 0, 10, 0, 0]

    dz = 0.1
    sgn = -1

    while True:

        image = vs.get_image(pose)

        if not VisionSensor.display_image(image, 'Image'):
            break

        pose[2] += sgn * dz

        if (pose[2] > 10):
            sgn = -1

        if (pose[2] < 5):
            sgn = +1


if __name__ == '__main__':

    main()
