'''
Multirotor Dynamics class

Should work for any simulator, vehicle, or operating system
 
Based on:
 
    @inproceedings{DBLP:conf/icra/BouabdallahMS04,
      author    = {Samir Bouabdallah and Pierpaolo Murrieri and Roland Siegwart},
      title     = {Design and Control of an Indoor Micro Quadrotor},
      booktitle = {Proceedings of the 2004 {IEEE} International Conference on Robotics and
                  Automation, {ICRA} 2004, April 26 - May 1, 2004, New Orleans, LA, {USA}},
      pages     = {4393--4398},
      year      = {2004},
      crossref  = {DBLP:conf/icra/2004},
      url       = {https:#doi.org/10.1109/ROBOT.2004.1302409},
      doi       = {10.1109/ROBOT.2004.1302409},
      timestamp = {Sun, 04 Jun 2017 01:00:00 +0200},
      biburl    = {https:#dblp.org/rec/bib/conf/icra/BouabdallahMS04},
      bibsource = {dblp computer science bibliography, https:#dblp.org}
    }
 
Copyright (C) 2019 Simon D. Levy
 
MIT License
'''

import numpy as np

class Parameters:
    '''
    Class for parameters from the table below Equation 3
    '''

    def __init__(self, b,  d,  m,  l,  Ix,  Iy,  Iz,  Jr, maxrpm):
   
        self.b = b
        self.d = d
        self.m = m
        self.l = l
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        self.Jr = Jr

        self.maxrpm = maxrpm

class MultirotorDynamics:
    '''
    Abstract class for multirotor dynamics.  You implementing class should define the following methods:

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
    STATE_X         = 0
    STATE_X_DOT     = 1
    STATE_Y         = 2
    STATE_Y_DOT     = 3
    STATE_Z         = 4
    STATE_Z_DOT     = 5
    STATE_PHI       = 6
    STATE_PHI_DOT   = 7
    STATE_THETA     = 8
    STATE_THETA_DOT = 9
    STATE_PSI       = 10
    STATE_PSI_DOT   = 11

    # universal constants
    g = 9.80665 # might want to allow this to vary!

    def __init__(self, params, motorCount):
        '''
        Constructor
        Initializes kinematic pose, with flag for whether we're airbone (helps with testing gravity).
        airborne allows us to start on the ground (default) or in the air (e.g., gravity test)
        '''
        self._p = params
        self._motorCount = motorCount

        self._omegas  = np.zeros(motorCount)

        # Always start at location (0,0,0) with zero velocities
        self._x    = np.zeros(12)
        self._dxdt = np.zeros(12)

        self._airborne = False

        # Values computed in Equation 6
        self._U1 = 0     # total thrust
        self._U2 = 0     # roll thrust right
        self._U3 = 0     # pitch thrust forward
        self._U4 = 0     # yaw thrust clockwise
        self._Omega = 0  # torque clockwise

        # Initialize inertial frame acceleration in NED coordinates
        self._inertialAccel = MultirotorDynamics._bodyZToInertiall(-MultirotorDynamics.g, (0,0,0))

    def setMotors(self, motorvals):
        '''
        Uses motor values to implement Equation 6.
        motorvals in interval [0,1]
        '''

        # Convert the  motor values to radians per second
        self._omegas = self._computeMotorSpeed(motorvals) #rad/s

        # Compute overall torque from omegas before squaring
        self._Omega = self.u4(self._omegas)

        # Overall thrust is sum of squared omegas
        omegas2 = self._omegas**2
        self._U1 = np.sum(self._p.b * omegas2)

        # Use the squared Omegas to implement the rest of Eqn. 6
        self._U2 = self._p.l * self._p.b * self.u2(omegas2)
        self._U3 = self._p.l * self._p.b * self.u3(omegas2)
        self._U4 = self._p.d * self.u4(omegas2)
        
    def update(self, dt):
        '''
        Updates state.
        dt time in seconds since previous update
        '''

        # Use the current Euler angles to rotate the orthogonal thrust vector into the inertial frame.
        # Negate to use NED.
        euler = ( self._x[6], self._x[8], self._x[10] )
        accelNED = MultirotorDynamics._bodyZToInertiall(-self._U1 / self._p.m, euler)

        # We're airborne once net downward acceleration goes below zero
        netz = accelNED[2] + MultirotorDynamics.g

        # If we're not airborne, we become airborne when downward acceleration has become negative
        if not self._airborne:
            self._airborne = netz < 0

        # Once airborne, we can update dynamics
        if self._airborne:

            # Compute the state derivatives using Equation 12
            self._computeStateDerivative(accelNED, netz)

            # Compute state as first temporal integral of first temporal derivative
            self._x += dt * self._dxdt

            # Once airborne, inertial-frame acceleration is same as NED acceleration
            self._inertialAccel = accelNED.copy()

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
        self._airborne = self._x[self.STATE_Z] < 0

    def _computeStateDerivative(self, accelNED, netz):
        '''
        Implements Equation 12 computing temporal first derivative of state.
        Should fill _dxdx[0..11] with appropriate values.
        accelNED acceleration in NED inertial frame
        netz accelNED[2] with gravitational constant added in
        phidot rotational acceleration in roll axis
        thedot rotational acceleration in pitch axis
        psidot rotational acceleration in yaw axis
        '''
 
        phidot = self._x[self.STATE_PHI_DOT]
        thedot = self._x[self.STATE_THETA_DOT]
        psidot = self._x[self.STATE_PSI_DOT]

        p = self._p

        self._dxdt[self.STATE_X]         = self._x[self.STATE_X_DOT]
        self._dxdt[self.STATE_X_DOT]     = accelNED[0]         
        self._dxdt[self.STATE_Y]         = self._x[self.STATE_Y_DOT]
        self._dxdt[self.STATE_Y_DOT]     = accelNED[1]         
        self._dxdt[self.STATE_Z]         = self._x[self.STATE_Z_DOT]
        self._dxdt[self.STATE_Z_DOT]     = netz                
        self._dxdt[self.STATE_PHI]       = phidot                                                                               
        self._dxdt[self.STATE_PHI_DOT]   = psidot * thedot * (p.Iy - p.Iz) / p.Ix - p.Jr / p.Ix * thedot * self._Omega + self._U2 / p.Ix    
        self._dxdt[self.STATE_THETA]     = thedot                                                                               
        self._dxdt[self.STATE_THETA_DOT] = -(psidot * phidot * (p.Iz - p.Ix) / p.Iy + p.Jr / p.Iy * phidot * self._Omega + self._U3 / p.Iy) 
        self._dxdt[self.STATE_PSI]       = psidot                                                                               
        self._dxdt[self.STATE_PSI_DOT]   = thedot * phidot * (p.Ix - p.Iy) / p.Iz + self._U4 / p.Iz                                   

    def _computeMotorSpeed(self, motorvals):
        '''
        Computes motor speed base on motor value
        motorval motor values in [0,1]
        return motor speed in rad/s
        '''
        return np.array(motorvals) * self._p.maxrpm * np.pi / 30

    def _bodyZToInertiall(bodyZ, rotation):
        '''
        _bodyToInertial method optimized for body X=Y=0
        '''
    
        cph, cth, cps, sph, sth, sps = MultirotorDynamics._sincos(rotation)

        # This is the rightmost column of the body-to-inertial rotation matrix
        R = np.array([sph * sps + cph * cps * sth, cph * sps * sth - cps * sph, cph * cth])

        return bodyZ * R

    def _inertialToBody(inertial, rotation):
    
        cph, cth, cps, sph, sth, sps = MultirotorDynamics._sincos(rotation)

        R = [[cps * cth,                    cth * sps,                         -sth],
             [cps * sph * sth - cph * sps,  cph * cps + sph * sps * sth,  cth * sph],
             [sph * sps + cph * cps * sth,  cph * sps * sth - cps * sph,  cph * cth]]

        return np.dot(R, inertial)

    def _bodyToInertial(body, rotation, inertial):
        '''
         Frame-of-reference conversion routines.
         
         See Section 5 of http:#www.chrobotics.com/library/understanding-euler-angles
        '''

        cph, cth, cps, sph, sth, sps = MultirotorDynamics._sincos(rotation)

        R = [[cps * cth,  cps * sph * sth - cph * sps,  sph * sps + cph * cps * sth],
             [cth * sps,  cph * cps + sph * sps * sth,  cph * sps * sth - cps * sph],
             [-sth,       cth * sph,                                      cph * cth]]

        return np.dot(R, body)

    def _eulerToQuaternion(euler):

        cph, cth, cps, sph, sth, sps = MultirotorDynamics._sincos(euler/2)

        return [[ cph * cth * cps + sph * sth * sps],
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


