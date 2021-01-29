#!/usr/bin/env python3
'''
Vision Sensor simulation

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np
# import matplotlib.pyplot as plt


class VisionSensor(object):

    def __init__(self, resolution, field_of_view, sensor_size):

        self.resolution = resolution
        self.sensor_size = sensor_size

        # Get focal length f from equations in
        # http://paulbourke.net/miscellaneous/lens/
        #
        # field of view = 2 atan(0.5 width / focallength)
        #
        # Therefore focalllength = width / (2 tan(field of view /2))
        #
        self.focal_length = (sensor_size /
                             (2 * np.tan(np.radians(field_of_view/2))))

    def get_image(self, points):
        '''
        @param Nx2 array of points
        @return self.resolution X self.resolution image
        '''

        # Convert to target indices
        j = (((points[:, 0] / 2 + 1) / 2 * self.resolution /
             self.sensor_size).astype(int))
        k = (((points[:, 1] / 2 + 1) / 2 * self.resolution /
             self.sensor_size).astype(int))

        # Use indices to populate image
        image = np.zeros((self.resolution, self.resolution)).astype('uint8')
        image[j, k] = 1

        return image


# End of VisionSensor class -------------------------------------------------


def main():

    return


if __name__ == '__main__':

    main()
