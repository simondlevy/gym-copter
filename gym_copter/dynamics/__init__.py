'''
Multirotor Dynamics class

Should work for any simulator, vehicle, or operating system

Based on:

    @inproceedings{DBLP:conf/icra/BouabdallahMS04,
      author    = {Samir Bouabdallah and Pierpaolo Murrieri and
                   Roland Siegwart},
      title     = {Design and Control of an Indoor Micro Quadrotor},
      booktitle = {Proceedings of the 2004 {IEEE} International Conference on
                  Robotics and Automation, {ICRA} 2004, April 26 - May 1, 2004,
                  New Orleans, LA, {USA}},
      pages     = {4393--4398},
      year      = {2004},
      crossref  = {DBLP:conf/icra/2004},
      url       = {https:#doi.org/10.1109/ROBOT.2004.1302409},
      doi       = {10.1109/ROBOT.2004.1302409},
      timestamp = {Sun, 04 Jun 2017 01:00:00 +0200},
      biburl    = {https:#dblp.org/rec/bib/conf/icra/BouabdallahMS04},
      bibsource = {dblp computer science bibliography, https:#dblp.org}
    }

Copyright (C) 2021 Simon D. Levy, Alex Sender

MIT License
'''

import numpy as np
import abc


class MultirotorDynamics:
    '''
    Abstract class for multirotor dynamics.  You implementing class should
    define the following methods:

        # roll right
        u2(omega^2)

        # pitch forward
        u3(omega^2)

        # yaw cw
        u4(omega^2)
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

    def __init__( self, vparams, framesPerSecond):
        '''
        Constructor initializes kinematic pose, with flag for whether we're
        airbone (helps with testing gravity).
        '''

        # World params, overridable by setWorldParams()
        self.g = 8.80665  # gravity
        self.rho = 1.225  # air density

        # Vehicle parameters [see Bouabdallah et al. 2004]
        self.D = vparams['D']     # drag coefficient
        self.M = vparams['M']     # mass
        self.Ix = vparams['Ix']   # moment of intertia X
        self.Iy = vparams['Iy']   # moment of intertia Y
        self.Iz = vparams['Iz']   # moment of intertia Z
        self.Jr = vparams['Jr']   # rotor inertia

        self.maxrpm = vparams['maxrpm']

        self._dt = 1. / framesPerSecond

        # Always start at location (0,0,0) with zero velocities
        self._x = np.zeros(12)
        self._dxdt = np.zeros(12)

        # Start on ground
        self._status = self.STATUS_LANDED

        # Values computed in Equation 6
        self._U1 = 0     # total thrust
        self._U2 = 0     # roll thrust right
        self._U3 = 0     # pitch thrust forward
        self._U4 = 0     # yaw thrust clockwise
        self._Omega = 0  # torque clockwise

        # Initialize inertial frame acceleration in NED coordinates
        self._inertialAccel = (
            MultirotorDynamics._bodyZToInertial(-self.g, (0, 0, 0)))

        # No perturbation yet
        self._perturb = np.zeros(6)

    def setWorldParams(self, g, rho):

        self.g = g
        self.rho = rho

    def setMotors(self, motorvals):
        '''
        Uses motor values to implement Equation 6.
        motorvals in interval [0,1]
        '''

        # Convert the  motor values to radians per second
        omegas = self._computeMotorSpeed(motorvals)

        # Compute individual motor thrusts are as air density times square of
        # motor speed
        omegas2 = self.rho * omegas**2

        # Overall thrust, as well as pitch and roll, depend on vehicle type
        self._U1, self._U2, self._U3 = self._getThrusts(np.sum(omegas2),
                                                        self._u2(omegas2),
                                                        self._u3(omegas2))

        # Compute yaw torque
        self._U4 = self.D * self._u4(omegas2)

    def update(self):
        '''
        Updates state.
        '''

        # Use the current Euler angles to rotate the orthogonal thrust vector
        # into the inertial frame.  Negate to use NED.
        euler = (self._x[6], self._x[8], self._x[10])
        accelNED = (MultirotorDynamics._bodyZToInertial(-self._U1 /
                    self.M, euler))

        # Compute net vertical acceleration by subtracting gravity
        netz = accelNED[2] + self.g

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
            self._computeStateDerivative(accelNED, netz)

            # Add instantaneous perturbation
            self._dxdt[1::2] += self._perturb

            # Compute state as first temporal integral of first temporal
            # derivative
            self._x += self._dt * self._dxdt

            # Once airborne, inertial-frame acceleration is same as NED
            # acceleration
            self._inertialAccel = accelNED.copy()

        # Reset instantaneous perturbation
        self._perturb = np.zeros(6)

    def getState(self):
        '''
        Returns a copy of the state vector as a tuple
        '''
        return tuple(self._x)

    def setState(self, state):
        '''
        Sets the state to the values specified in a sequence
        '''
        self._x = np.array(state)
        self._status = (self.STATUS_AIRBORNE
                        if self._x[self.STATE_Z] < 0
                        else self.STATUS_LANDED)

    def getStatus(self):

        return self._status

    def perturb(self, force):

        self._perturb = force / self.M

    def _computeStateDerivative(self, accelNED, netz):
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

        self._dxdt[self.STATE_X_DOT] = accelNED[0] + self._perturb[0]

        self._dxdt[self.STATE_Y] = self._x[self.STATE_Y_DOT]

        self._dxdt[self.STATE_Y_DOT] = accelNED[1] + self._perturb[1]

        self._dxdt[self.STATE_Z] = self._x[self.STATE_Z_DOT]

        self._dxdt[self.STATE_Z_DOT] = netz + self._perturb[2]

        self._dxdt[self.STATE_PHI] = phidot

        self._dxdt[self.STATE_PHI_DOT] = (
            psidot*thedot*(self.Iy-self.Iz) / self.Ix-self.Jr /
            self.Ix*thedot*self._Omega + self._U2 / self.Ix + self._perturb[3])

        self._dxdt[self.STATE_THETA] = thedot

        self._dxdt[self.STATE_THETA_DOT] = (
                -(psidot*phidot*(self.Iz-self.Ix) / self.Iy + self.Jr /
                  self.Iy*phidot*self._Omega + self._U3 / self.Iy) +
                self._perturb[4])

        self._dxdt[self.STATE_PSI] = psidot

        self._dxdt[self.STATE_PSI_DOT] = (
            thedot*phidot*(self.Ix-self.Iy)/self.Iz +
            self._U4/self.Iz + self._perturb[5])

    def _computeMotorSpeed(self, motorvals):
        '''
        Computes motor speed base on motor value
        motorval motor values in [0,1]
        return motor speed in rad/s
        '''
        return np.array(motorvals) * self.maxrpm * np.pi / 30

    def _bodyZToInertial(bodyZ, rotation):
        '''
        _bodyToInertial method optimized for body X=Y=0
        '''

        cph, cth, cps, sph, sth, sps = MultirotorDynamics._sincos(rotation)

        # This is the rightmost column of the body-to-inertial rotation matrix
        R = np.array([sph*sps+cph*cps*sth, cph*sps*sth-cps*sph, cph*cth])

        return bodyZ * R

    def _inertialToBody(inertial, rotation):

        cph, cth, cps, sph, sth, sps = MultirotorDynamics._sincos(rotation)

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

        cph, cth, cps, sph, sth, sps = MultirotorDynamics._sincos(rotation)

        R = [[cps*cth, cps*sph*sth-cph*sps, sph*sps + cph*cps*sth],
             [cth*sps, cph*cps+sph*sps*sth, cph*sps*sth-cps*sph],
             [-sth, cth*sph, cph*cth]]

        return np.dot(R, body)

    def _eulerToQuaternion(euler):

        cph, cth, cps, sph, sth, sps = MultirotorDynamics._sincos(euler/2)

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

    @abc.abstractmethod
    def _getThrusts(self, u1, u2, u3):
        '''
        Depends on vehicle type (fixed-pitch blades or collective pitch)
        '''
        return

    @abc.abstractmethod
    def _u2(self,  o):
        '''
        Roll right. Depends on vehicle type (fixed-pitch blades or collective
        pitch) and layout (quad, hex, etc.)
        '''
        return

    @abc.abstractmethod
    def _u3(self,  o):
        '''
        Pitch forward. Depends on vehicle type (fixed-pitch blades or
        collective pitch) and layout (quad, hex, etc.)
        '''
        return

    @abc.abstractmethod
    def _u4(self,  o):
        '''
        Yaw clockwise. Depends on vehicle type (fixed-pitch blades or
        collective pitch) and layout (quad, hex, etc.)
        '''
        return
