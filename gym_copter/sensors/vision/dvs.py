#!/usr/bin/env python3
'''
Dynamic Vision Sensor simulation

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np
import cv2

from gym_copter.sensors.vision.vs import VisionSensor


class DVS(VisionSensor):

    def __init__(self, objsize=1, res=128, fov=60):
        '''
        @param size size meters
        @param res resolution in (pixels)
        @param fov field of view (degrees)
        '''

        VisionSensor.__init__(self, objsize, res)

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

    def display_image(self, image, name='Vision', display_size=400):
        '''
        Scale up and display the image
        '''
        # Image comes in as greyscale quantized to -1, 0, +1.  We convert it to 
        # color.
        image = cv2.resize(image, ((display_size, )*2))
        cimage = np.zeros((display_size, display_size, 3)).astype('uint8')
        r, c = np.where(image == -1)
        cimage[r,c,2] = 255
        r, c = np.where(image == +1)
        cimage[r,c,1] = 255
        cv2.imshow(name, cimage)
        cv2.moveWindow(name, 725, 0);
        return cv2.waitKey(10) != 27  # ESC

# End of DVS class -------------------------------------------------


def main():

    XRANGE = 4
    SPEED = .02

    # Arbitrary object size (1m)
    dvs = DVS(1)

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
