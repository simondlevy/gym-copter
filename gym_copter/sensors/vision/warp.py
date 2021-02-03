#!/usr/bin/env python3
'''
Use perspective warping on a camera image

Adapted from

  https:# stackoverflow.com/questions/17087446/

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
'''

import argparse
from argparse import ArgumentDefaultsHelpFormatter
import cv2
import numpy as np


def hypot(shape):
    return np.sqrt(shape[0]**2 + shape[1]**2)


def getWarpMatrix(sz, psi, theta, phi, fovy):

    st = np.sin(np.radians(psi))
    ct = np.cos(np.radians(psi))
    sp = np.sin(np.radians(theta))
    cp = np.cos(np.radians(theta))
    sg = np.sin(np.radians(phi))
    cg = np.cos(np.radians(phi))

    halfFovy = fovy*0.5
    d = hypot(sz)
    sideLength = d/np.cos(np.radians(halfFovy))
    h = d/(2.0*np.sin(np.radians(halfFovy)))
    n = h-(d/2.0)
    f = h+(d/2.0)

    F = np.zeros((4, 4))  # 4x4 transformation matrix F

    Rpsi = np.eye(4)      # 4x4 rotation matrix around Z-axis by psi degrees
    Rtheta = np.eye(4)    # 4x4 rotation matrix around X-axis by theta degrees
    Rphi = np.eye(4)      # 4x4 rotation matrix around Y-axis by phi degrees
    T = np.eye(4)         # 4x4 translation matrix along Z-axis by -h units
    P = np.zeros((4, 4))  # Allocate 4x4 projection matrix

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
    P[0, 0] = P[1, 1] = 1.0/np.tan(np.radians(halfFovy))
    P[2, 2] = -(f+n)/(f-n)
    P[2, 3] = -(2.0*f*n)/(f-n)
    P[3, 2] = -1.0

    # Compose transformations
    F = np.dot(np.dot(np.dot(np.dot(P, T), Rtheta), Rpsi), Rphi)

    # Transform 4x4 points
    halfW = sz[1]/2
    halfH = sz[0]/2

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
        ptsOutPt2f[i] = (ptsOutMat[i, 0, :2] + np.ones(2)) * (sideLength * 0.5)

    return cv2.getPerspectiveTransform(ptsInPt2f, ptsOutPt2f)


def warpImage(src, x, y, z, phi, theta, psi, fovy, size):

    halfFovy = fovy*0.5
    d = hypot(src.shape)
    sideLength = int(d/np.cos(np.radians(halfFovy)))

    # Compute warp matrix
    M = getWarpMatrix(src.shape, psi, theta, phi, fovy)

    # Do actual image war0
    return cv2.warpPerspective(src, M, (sideLength, sideLength))


def main():

    parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--file',  default='chevron.png', help='Input file')
    parser.add_argument('--x',  type=float, default=0, help='X coordinate (m)')
    parser.add_argument('--y',  type=float, default=0, help='Y coordinate (m)')
    parser.add_argument('--z',  type=float, default=5, help='Z coordinate (m)')
    parser.add_argument('--phi',  type=float, default=0,
                        help='Roll angle (deg)')
    parser.add_argument('--theta',  type=float, default=0,
                        help='Pitch angle (deg)')
    parser.add_argument('--psi',  type=float, default=0,
                        help='Yaw angle (deg)')
    parser.add_argument('--fov',  type=float, default=30,
                        help='Field of view (deg)')
    parser.add_argument('--size',  type=float, default=1,
                        help='Object size (m)')

    args = parser.parse_args()

    image = cv2.imread(args.file)[:, :, 0]

    while(True):

        warped = warpImage(image,
                           args.x,
                           args.y,
                           args.z,
                           args.phi,
                           args.theta,
                           args.psi,
                           args.fov,
                           args.size)

        cv2.imshow(args.file, warped)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':

    main()
