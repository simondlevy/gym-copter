#!/usr/bin/env python3
'''
Vision Sensor simulation

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import numpy as np
import cv2


class VisionSensor(object):

    def __init__(self, object_size,
                 resolution=128,
                 field_of_view=60,
                 sensor_size=8):
        '''
        @param object_size meters
        @param resolution pixels
        @param field_of_view degrees
        @param sensor_size millimeters
        '''

        self.object_size = object_size
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

    def get_image(self, pose):

        image = np.zeros((self.resolution, )*2)
        cv2.circle(image, (64, 64), 10, 1, thickness=1)
        return image


class DVS(VisionSensor):

    def __init__(self, object_size,
                 resolution=128,
                 field_of_view=60,
                 sensor_size=8):
        '''
        @param object_size meters
        @param resolution pixels
        @param field_of_view degrees
        @param sensor_size millimeters
        '''

        VisionSensor.__init__(self, object_size, resolution,
                              field_of_view, sensor_size)

        self.image_prev = None

    def get_events(self, pose):
        '''
        @param pos x,y,z,phi,theta
        @return list of x,y sensor events
        '''

        # Use altitude as distance to object
        # u = pose[2]

        # Get image magnification m from equations in
        # https://www.aplustopper.com/numerical-methods-in-lens/
        #
        # 1/u + 1/v = 1/f
        #
        # m = v/u
        #
        # Therefore m = 1 / (u/f - 1)
        #
        # m = 1 / (u / self.focal_length - 1)

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

    # Arbitrary object size (2m)
    dvs = DVS(2)

    # Arbitrary stating pose
    pose = [10, 0, 10]

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

    vs = VisionSensor(2)

    # Arbitrary stating position
    pose = [10, 0, 10]

    while True:

        image = vs.get_image(pose)

        if not display_image(image, 'Image'):
            break


if __name__ == '__main__':

    vision()
