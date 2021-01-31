#!/usr/bin/env python3
'''
Vision Sensor simulation

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np
import cv2

from gym_copter.sensors.vision.vs import VisionSensor


class DVS(VisionSensor):

    def __init__(self, object_size, resolution=128, field_of_view=60):
        '''
        @param object_size meters
        @param resolution pixels
        @param field_of_view degrees
        '''

        VisionSensor.__init__(self, object_size, resolution)

        self.image_prev = None

    def get_events(self, pose):
        '''
        @param pos x,y,z,phi,theta
        @return list of x,y sensor events
        '''

        # Make an image of an arbitrarily-shaped object
        image_curr = self._make_image(pose)

        # First time around, no eventsimage.  Subsequent times, do a first
        # difference to get the events.
        image_diff = (self.image_prev - image_curr
                      if self.image_prev is not None
                      else np.zeros((self.resolution, self.resolution)))

        # Track previous event image for first difference
        self.image_prev = image_curr

        # Collect and return nonzero points
        return [(x, y, image_diff[x, y])
                for x, y in zip(*np.nonzero(image_diff))]

    def _make_image(self, pose):

        image = np.zeros((self.resolution, self.resolution)).astype('int8')

        cv2.circle(image, (pose[0], pose[1]), 10, 1, thickness=-1)

        return image

# End of VisionSensor classes -------------------------------------------------


def main():

    # Arbitrary object size (1m)
    dvs = DVS(1)

    # Arbitrary stating pose
    pose = [10, 0, 10, 0, 0]

    while True:

        # Get events
        events = dvs.get_events(pose)

        # Make an image from the events
        image = np.zeros((128, 128, 3)).astype('uint8')
        for x, y, p in events:
            image[x][y][1 if p == +1 else 2] = 255

        if not VisionSensor.display_image(image, 'Events'):
            break

        # Move pose across field of view
        pose[0] = (pose[0] + 1) % 128


if __name__ == '__main__':

    main()
