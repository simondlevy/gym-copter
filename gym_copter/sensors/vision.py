#!/usr/bin/env python3
'''
Vision Sensor simulation

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np
# import matplotlib.pyplot as plt


class VisionSensor(object):

    def __init__(self, resolution=128, field_of_view=60, sensor_size=8):
        '''
        @param resolution pixels
        @param field_of_view degrees
        @param sensor_size millimeters
        '''

        self.resolution = resolution
        self.sensor_size = sensor_size / 1000  # mm to m

        # Get focal length f from equations in
        # http://paulbourke.net/miscellaneous/lens/
        #
        # field of view = 2 atan(0.5 width / focallength)
        #
        # Therefore focalllength = width / (2 tan(field of view /2))
        #
        self.focal_length = (self.sensor_size /
                             (2 * np.tan(np.radians(field_of_view/2))))

    def get_image(self, points, distance):
        '''
        @param Nx2 array of points
        @param distance from points
        @return self.resolution X self.resolution image
        '''

        # Get image magnification m from equations in
        # https://www.aplustopper.com/numerical-methods-in-lens/
        #
        # 1/u + 1/v = 1/f
        #
        # m = v/u
        #
        # Therefore m = 1 / (u/f - 1)
        #
        # m = 1 / (distance / self.focal_length - 1)

        # Convert to target indices
        # j = (((points[:, 0] / 2 + 1) / 2 * self.resolution /
        #     self.sensor_size).astype(int))
        # k = (((points[:, 1] / 2 + 1) / 2 * self.resolution /
        #      self.sensor_size).astype(int))

        # Use indices to populate image
        image = np.zeros((self.resolution, self.resolution)).astype('uint8')
        image[:100, 50] = 1
        # image[j, k] = 1

        return image


# End of VisionSensor class -------------------------------------------------


def main():

    return


if __name__ == '__main__':

    main()
