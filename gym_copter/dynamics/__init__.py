'''
Multirotor Dynamics class

Should work for any simulator, vehicle, or operating system

Copyright (C) 2023 Simon D. Levy, Alex Sender

MIT License
'''

import numpy as np


class Dynamics:
    '''
    Dynamics class for quad-X frames using ArduPilot motor layout:

    3cw   1ccw

        ^

    2ccw  4cw

    '''

    '''
    Position map for state vector
    '''
    (STATE_X,
     STATE_X_DOT,
     STATE_Y,
     STATE_Y_DOT,
     STATE_Z,
     STATE_Z_DOT,
     STATE_PHI,
     STATE_PHI_DOT,
     STATE_THETA,
     STATE_THETA_DOT,
     STATE_PSI,
     STATE_PSI_DOT) = range(12)

    '''
    Flight status: LANDED -> AIRBORNE -> CRASHED or
                   LANDED -> AIRBORNE -> LEVELING -> LANDED
    '''
    (STATUS_CRASHED,
     STATUS_LANDED,
     STATUS_LEVELING,
     STATUS_AIRBORNE) = range(4)

    # Safe landing criteria
    LANDING_VEL_X = 2.0
    LANDING_VEL_Y = 1.0
    LANDING_ANGLE = np.pi/4

    # Graviational constant
    G = 9.80665

    def __init__(self, params, framesPerSecond):

        '''
        Constructor initializes kinematic pose, with flag for whether we're
        airbone (helps with testing gravity).
        '''

        # Vehicle parameters [see Bouabdallah et al. 2004]
        self.D = params['D']     # drag coefficient
        self.M = params['M']     # mass
        self.Ix = params['Ix']   # moment of intertia X
        self.Iy = params['Iy']   # moment of intertia Y
        self.Iz = params['Iz']   # moment of intertia Z
        self.Jr = params['Jr']   # rotor inertia
        self.B = params['B']     # thrust coefficient
        self.L = params['L']     # arm length

        self.maxrpm = params['maxrpm']

        self._dt = 1. / framesPerSecond
        self._ticks = 0

        # Always start at location (0,0,0) with zero velocities
        self._x = np.zeros(12)
        self._dxdt = np.zeros(12)

        # Start on ground
        self._status = self.STATUS_LANDED

        # Initialize inertial frame acceleration in NED coordinates
        self._inertialAccel = (
            Dynamics._bodyZToInertial(-self.G, (0, 0, 0)))

    def setMotors(self, motorvals):
        '''
        Implements Equations 6 and 12 from Bouabdallah et al. (2004)
        '''

        # Convert the  motor values to radians per second
        omegas = np.array(motorvals) * self.maxrpm * np.pi / 30

        # Compute individual motor thrusts are as air density times square of
        # motor speed
        omegas2 = omegas**2

        # Compute overall thrust, plus roll and pitch
        U1 = self.B * np.sum(omegas2)
        U2 = self.L * self.B * self._u2(omegas2)
        U3 = self.L * self.B * self._u3(omegas2)

        # Compute yaw torque
        U4 = self.D * self._u4(omegas2)

        # Ignore Omega ("disturbance") part of Equation 6 for now
        Omega = 0

        # Use the current Euler angles to rotate the orthogonal thrust vector
        # into the inertial frame.  Negate to use NED.
        euler = (self._x[6], self._x[8], self._x[10])
        accelNED = Dynamics._bodyZToInertial(-U1 / self.M, euler)

        # Compute net vertical acceleration by subtracting gravity
        netz = accelNED[2] + self.G

        # If we're not airborne, we become airborne when downward acceleration
        # has become negative
        if self._status == self.STATUS_LANDED:
            if netz < 0:
                self._status = self.STATUS_AIRBORNE

        # Leveling mode: change roll, pitch angles for  rendering
        if self._status == self.STATUS_LEVELING:

            self._x[self.STATE_PHI] = 0
            self._x[self.STATE_THETA] = 0
            self._status = self.STATUS_LANDED

        # Once airborne, we can update dynamics
        elif self._status == self.STATUS_AIRBORNE:

            # If we've descended to the ground
            if self._x[self.STATE_Z] > 0 and self._x[self.STATE_Z_DOT] > 0:

                # Big angles indicate a crash
                phi = self._x[self.STATE_PHI]
                velx = self._x[self.STATE_Y_DOT]
                vely = self._x[self.STATE_Z_DOT]
                if (vely > self.LANDING_VEL_Y or
                   abs(velx) > self.LANDING_VEL_X or
                   abs(phi) > self.LANDING_ANGLE):
                    self._status = self.STATUS_CRASHED

                # Small angles indicate leveling
                else:
                    self._status = self.STATUS_LEVELING

                return

            # Compute the state derivatives using Equation 12
            self._computeStateDerivative(accelNED, netz, U2, U3, U4, Omega)

            # Compute state as first temporal integral of first temporal
            # derivative
            self._x += self._dt * self._dxdt

            # Once airborne, inertial-frame acceleration is same as NED
            # acceleration
            self._inertialAccel = accelNED.copy()

        # Update time
        self._ticks += 1

    def getState(self):
        '''
        Returns the state vector as a dictionary
        '''
        keys = ('x', 'dx', 'y', 'dy', 'z', 'dz',
                'phi', 'dphi', 'theta', 'dtheta', 'psi', 'dpsi')

        return {key: value for key, value in zip(keys, self._x)}

    def _u2(self,  o):
        '''
        roll right
        '''
        return (o[1] + o[2]) - (o[0] + o[3])

    def _u3(self,  o):
        '''
        pitch forward
        '''
        return (o[1] + o[3]) - (o[0] + o[2])

    def _u4(self,  o):
        '''
        yaw cw
        '''
        return (o[0] + o[1]) - (o[2] + o[3])

    def _computeStateDerivative(self, accelNED, netz, U2, U3, U4, Omega):
        '''
        Implements Equation 12 computing temporal first derivative of state.
        Should fill _dxdx[0..11] with appropriate values.
        accelNED acceleration in NED inertial frame
        netz accelNED[2] with gravitational constant added in
        '''

        phidot = self._x[self.STATE_PHI_DOT]
        thedot = self._x[self.STATE_THETA_DOT]
        psidot = self._x[self.STATE_PSI_DOT]

        self._dxdt[self.STATE_X] = self._x[self.STATE_X_DOT]

        self._dxdt[self.STATE_X_DOT] = accelNED[0]

        self._dxdt[self.STATE_Y] = self._x[self.STATE_Y_DOT]

        self._dxdt[self.STATE_Y_DOT] = accelNED[1]

        self._dxdt[self.STATE_Z] = self._x[self.STATE_Z_DOT]

        self._dxdt[self.STATE_Z_DOT] = netz

        self._dxdt[self.STATE_PHI] = phidot

        self._dxdt[self.STATE_PHI_DOT] = (
            psidot*thedot*(self.Iy-self.Iz) / self.Ix-self.Jr /
            self.Ix*thedot*Omega + U2 / self.Ix)

        self._dxdt[self.STATE_THETA] = thedot

        self._dxdt[self.STATE_THETA_DOT] = (
                -(psidot*phidot*(self.Iz-self.Ix) / self.Iy + self.Jr /
                  self.Iy*phidot*Omega + U3 / self.Iy))

        self._dxdt[self.STATE_PSI] = psidot

        self._dxdt[self.STATE_PSI_DOT] = (
            thedot*phidot*(self.Ix-self.Iy)/self.Iz + U4/self.Iz)

    def _bodyZToInertial(bodyZ, rotation):
        '''
        _bodyToInertial method optimized for body X=Y=0
        '''

        cph, cth, cps, sph, sth, sps = Dynamics._sincos(rotation)

        # This is the rightmost column of the body-to-inertial rotation matrix
        R = np.array([sph*sps+cph*cps*sth, cph*sps*sth-cps*sph, cph*cth])

        return bodyZ * R

    def _inertialToBody(inertial, rotation):

        cph, cth, cps, sph, sth, sps = Dynamics._sincos(rotation)

        R = [[cps*cth, cth*sps, -sth],
             [cps*sph*sth-cph*sps, cph*cps+sph*sps*sth, cth*sph],
             [sph*sps+cph*cps*sth, cph*sps*sth-cps*sph, cph*cth]]

        return np.dot(R, inertial)

    def _bodyToInertial(body, rotation, inertial):
        '''
         Frame-of-reference conversion routines.

         See Section 5 of
           http:www.chrobotics.com/library/understanding-euler-angles
        '''

        cph, cth, cps, sph, sth, sps = Dynamics._sincos(rotation)

        R = [[cps*cth, cps*sph*sth-cph*sps, sph*sps + cph*cps*sth],
             [cth*sps, cph*cps+sph*sps*sth, cph*sps*sth-cps*sph],
             [-sth, cth*sph, cph*cth]]

        return np.dot(R, body)

    def _eulerToQuaternion(euler):

        cph, cth, cps, sph, sth, sps = Dynamics._sincos(euler/2)

        return [[cph * cth * cps + sph * sth * sps],
                [cph * sth * sps - sph * cth * cps],
                [-cph * sth * cps - sph * cth * sps],
                [cph * cth * sps - sph * sth * cps]]

    def _sincos(angles):

        phi, the, psi = angles

        cph = np.cos(phi)
        cth = np.cos(the)
        cps = np.cos(psi)
        sph = np.sin(phi)
        sth = np.sin(the)
        sps = np.sin(psi)

        return cph, cth, cps, sph, sth, sps
