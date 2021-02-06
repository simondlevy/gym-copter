#!/usr/bin/env python3
'''
Vision Sensor simulation

Copyright (C) 2021 Simon D. Levy

Perspective warping adapted from https:# stackoverflow.com/questions/17087446/

MIT License
'''

import argparse
from argparse import ArgumentDefaultsHelpFormatter
import numpy as np
import cv2


class VisionSensor(object):

    def __init__(self, objsize=1, res=128, fov=30):
        '''
        @param size size meters
        @param res resolution in (pixels)
        @param fov field of view (degrees)
        '''

        self.objsize = objsize
        self.res = res
        self.fov = fov

    def getImage(self, x, y, z, phi, theta, psi):
        '''
        @param x, y, z position (meters)
        @param phi, theta, psi Euler angles (degrees)
        '''

        image = np.zeros((self.res, )*2)

        # Compute image center in pixels
        cx = self._locate(z, x)
        cy = self._locate(z, y)

        # Add a shape of your choice to the image
        self._add_shape(image, cx, cy, z)

        # Compute warp matrix
        M = self._getWarpMatrix(image.shape, psi, theta, phi)

        # Compute new image size for warping
        halfFov = self.fov/2
        d = VisionSensor._hypot(image.shape)
        sideLength = int(d/np.cos(np.radians(halfFov)))

        # Warp image
        warped = cv2.warpPerspective(image, M, (sideLength, sideLength))

        # Remove margin introduced by warping
        margin = (warped.shape[0] - image.shape[0]) // 2
        return warped[margin:-margin, margin:-margin]

    @staticmethod
    def display_image(image, name='Vision', display_size=400):
        '''
        Scale up and display the image
        '''
        image = cv2.resize(image, ((display_size, )*2))
        cv2.imshow(name, image)
        cv2.moveWindow(name, 725, 0);
        return cv2.waitKey(10) != 27  # ESC

    def _add_shape(self, image, cx, cy, z):

        # Scale radius by altitude
        r = self._scale(z, self.objsize)

        try:
            cv2.circle(image, (cx, cy), r, (255, 255, 255))

        except Exception:
            pass

    def _add_shape_pentagon(self, image, cx, cy, z):

        # A pentagon centered around the origin with unit width and height
        shape = np.array([[-.5, -0.0714],
                          [0, -0.5],
                          [+.5, -0.0714],
                          [+.25, +.5],
                          [-.25, +.5]])

        # Scale up the pentagon and center it in the image
        shape *= self._scale(z, self.objsize)
        shape[:, 0] += cx
        shape[:, 1] += cy

        # Draw the shapegon as a filled polygon
        cv2.fillPoly(image, [shape.astype('int32')], 255)

    def _locate(self, z, coord):

        return self._scale(z, coord) + self.res//2

    def _scale(self, z, val):

        return int(val * self.res / (2 * z * np.tan(np.radians(self.fov/2))))

    def _getWarpMatrix(self, size, psi, theta, phi):

        st = np.sin(np.radians(psi))
        ct = np.cos(np.radians(psi))
        sp = np.sin(np.radians(theta))
        cp = np.cos(np.radians(theta))
        sg = np.sin(np.radians(phi))
        cg = np.cos(np.radians(phi))

        halfFov = self.fov/2
        d = VisionSensor._hypot(size)
        sideLength = d/np.cos(np.radians(halfFov))
        h = d/(2.0*np.sin(np.radians(halfFov)))
        n = h-(d/2.0)
        f = h+(d/2.0)

        # 4x4 transformation matrix F
        F = np.zeros((4, 4))

        # 4x4 rotation matrix around Z-axis by psi degrees
        Rpsi = np.eye(4)

        # 4x4 rotation matrix around X-axis by theta degrees
        Rtheta = np.eye(4)

        # 4x4 rotation matrix around Y-axis by phi degrees
        Rphi = np.eye(4)

        # 4x4 translation matrix along Z-axis by -h units
        T = np.eye(4)

        # 4x4 projection matrix
        P = np.zeros((4, 4))

        # Rpsi
        Rpsi[0, 0] = Rpsi[1, 1] = ct
        Rpsi[0, 1] = -st
        Rpsi[1, 0] = st

        # Rtheta
        Rtheta[1, 1] = Rtheta[2, 2] = cp
        Rtheta[1, 2] = -sp
        Rtheta[2, 1] = sp

        # Rphi
        Rphi[0, 0] = Rphi[2, 2] = cg
        Rphi[0, 2] = -sg
        Rphi[2, 0] = sg

        # T
        T[2, 3] = -h

        # P
        P[0, 0] = P[1, 1] = 1.0/np.tan(np.radians(halfFov))
        P[2, 2] = -(f+n)/(f-n)
        P[2, 3] = -(2.0*f*n)/(f-n)
        P[3, 2] = -1.0

        # Compose transformations
        F = np.dot(np.dot(np.dot(np.dot(P, T), Rtheta), Rpsi), Rphi)

        # Transform 4x4 points
        halfW = size[1]/2
        halfH = size[0]/2

        ptsIn = np.array([-halfW, halfH, 0,
                          halfW, halfH, 0,
                          halfW, -halfH, 0,
                          -halfW, -halfH, 0])

        ptsInMat = np.reshape(ptsIn, (4, 1, 3))

        ptsOutMat = cv2.perspectiveTransform(ptsInMat, F)  # Transform points

        ptsInPt2f = np.zeros((4, 2)).astype('float32')
        ptsOutPt2f = np.zeros((4, 2)).astype('float32')

        for i in range(4):
            ptsInPt2f[i] = ptsInMat[i, 0, :2] + np.array([halfW, halfH])
            ptsOutPt2f[i] = ((ptsOutMat[i, 0, :2] +
                             np.ones(2)) * (sideLength * 0.5))

        return cv2.getPerspectiveTransform(ptsInPt2f, ptsOutPt2f)

    @staticmethod
    def _hypot(shape):
        return np.sqrt(shape[0]**2 + shape[1]**2)


# End of VisionSensor classes -------------------------------------------------


def main():

    parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--res',  type=int, default=128,
                        help='Pixel resolution')
    parser.add_argument('--fov',  type=float, default=30,
                        help='Field of view (deg)')
    parser.add_argument('--objsize',  type=float, default=1,
                        help='Object size (m)')
    parser.add_argument('--x',  type=float, default=0, help='X coordinate (m)')
    parser.add_argument('--y',  type=float, default=0, help='Y coordinate (m)')
    parser.add_argument('--z',  type=float, default=5, help='Z coordinate (m)')
    parser.add_argument('--phi',  type=float, default=0,
                        help='Roll angle (deg)')
    parser.add_argument('--theta',  type=float, default=0,
                        help='Pitch angle (deg)')
    parser.add_argument('--psi',  type=float, default=0,
                        help='Yaw angle (deg)')

    args = parser.parse_args()

    vs = VisionSensor(args.objsize, args.res, args.fov)

    image = vs.getImage(args.x, args.y, args.z,
                        args.phi, args.theta, args.psi)

    while True:

        if not VisionSensor.display_image(image):
            break


if __name__ == '__main__':

    main()
