#!/usr/bin/env python3
'''
Dynamic Vision Sensor simulation

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np
import cv2


class DVS:

    RADIUS = 1

    def __init__(self, resolution=128, field_of_view=60, sensor_size=8):
        '''
        @param resolution pixels
        @param field_of_view degrees
        @param sensor_size millimeters
        '''

        self.resolution = resolution
        self.sensor_size = sensor_size / 1000  # mm to m

        self.image_prev = None

        # Get focal length f from equations in
        # http://paulbourke.net/miscellaneous/lens/
        #
        # field of view = 2 atan(0.5 width / focallength)
        #
        # Therefore focalllength = width / (2 tan(field of view /2))
        #
        self.focal_length = (self.sensor_size /
                             (2 * np.tan(np.radians(field_of_view/2))))

    def get_image(self, pos):

        # Use altitude as distance to object
        u = pos[2]

        # Get image magnification m from equations in
        # https://www.aplustopper.com/numerical-methods-in-lens/
        #
        # 1/u + 1/v = 1/f
        #
        # m = v/u
        #
        # Therefore m = 1 / (u/f - 1)
        #
        m = 1 / (u / self.focal_length - 1)

        print(m)

        self.pos = pos

        image_curr = self._make_image(pos)

        # First time around, return an empty image.  Subsequent times, do a
        # first difference to get the image.
        image_diff = (self.image_prev - image_curr
                      if self.image_prev is not None
                      else np.zeros((self.resolution, self.resolution)))

        self.image_prev = image_curr

        return [(x, y, image_diff[x, y])
                for x, y in zip(*np.nonzero(image_diff))]

    def _make_image(self, pos):

        image = np.zeros((self.resolution, self.resolution)).astype('int8')

        cv2.circle(image, (pos[0], pos[1]), 10, 1, thickness=-1)

        return image

# End of DVS class -------------------------------------------------


def main():

    SCALEUP = 4

    dvs = DVS()

    pos = [64, 64, 10]

    while True:

        image = np.zeros((128, 128, 3)).astype('uint8')

        events = dvs.get_image(pos)

        for x, y, p in events:
            image[x][y][1 if p == +1 else 2] = 255

        # Scale up and display the image
        image = cv2.resize(image, (128*SCALEUP, 128*SCALEUP))
        cv2.imshow('Events', image)
        if cv2.waitKey(10) == 27:  # ESC
            break

        # Move position across field of view
        pos[0] = (pos[0] + 1) % 128


if __name__ == '__main__':

    main()
