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
        _, _, z, _hi, _heta = pose

        # Use trig formula to compute fraction of object in
        # current field of view.
        s = self.object_size / (2 * z * self.tana)

        image = np.zeros((self.resolution, )*2)

        # XXX Ignore effects of all but altitude for now
        cx, cy = (self.resolution//2, )*2

        # Add a circle with radius proportional to fraction of object
        # in current field of view.
        cv2.circle(image, (cx, cy), int(s*self.resolution), 1, thickness=1)

        return image

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

    dz = .1
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
