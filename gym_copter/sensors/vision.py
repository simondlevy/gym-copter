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

        # Use trig formula to compute fraction of object in
        # current field of view.
        s = self.object_size / (2 * pose[2] * self.tana)

        image = np.zeros((self.resolution, )*2)

        # XXX Ignore effects of all but altitude for now
        x, y = (self.resolution//2, )*2

        # Add a circle with radius proportional to fraction of object
        # in current field of view.
        cv2.circle(image, (x, y), int(s*self.resolution), 1, thickness=1)

        return image


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


def display_image(image, name, scaleup=4):
    '''
    Scale up and display the image
    '''
    image = cv2.resize(image, ((128*scaleup, )*2))
    cv2.imshow(name, image)
    return cv2.waitKey(10) != 27  # ESC


def dvs():

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

        if not display_image(image, 'Events'):
            break

        # Move pose across field of view
        pose[0] = (pose[0] + 1) % 128


def vision():

    vs = VisionSensor(1)

    # Arbitrary stating position
    pose = [0, 0, 10, 0, 0]

    dz = .1
    sgn = -1

    while True:

        image = vs.get_image(pose)

        if not display_image(image, 'Image'):
            break

        pose[2] += sgn * dz

        if (pose[2] > 10):
            sgn = -1

        if (pose[2] < 5):
            sgn = +1


if __name__ == '__main__':

    vision()
