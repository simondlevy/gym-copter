'''

Multirotor Dynamics class

  Header-only code for platform-independent multirotor dynamics
 
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

class MultirotorDynamics:

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

    class Pose:
        '''
        Exported pose representation
        '''
        def __init__(self):
            self.location = np.zeros(3)
            self.rotation = np.zeros(3)

    class State:
        '''
        Exported state representation
        '''
        def __init__(self):
            self.pose = MultirotorDynamics.Pose()
            self.angularVel  = np.zeros(3)
            self.bodyAccel   = np.zeros(3)
            self.inertialVel = np.zeros(3)
            self.quaternion  = np.zeros(4)

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

    # bodyToInertial method optimized for body X=Y=0
    def bodyZToInertial(bodyZ, rotation):
    
        phi, theta, psi = rotation

        cph = np.cos(phi)
        sph = np.sin(phi)
        cth = np.cos(theta)
        sth = np.sin(theta)
        cps = np.cos(psi)
        sps = np.sin(psi)

        # This is the rightmost column of the body-to-inertial rotation matrix
        R = ( sph * sps + cph * cps * sth,
              cph * sps * sth - cps * sph,
              cph * cth )

        return (bodyZ * R[i] for i in range(3))

    def inertialToBody(inertial, rotation, body):
    
        phi, theta, psi = rotation

        cph = np.cos(phi)
        sph = np.sin(phi)
        cth = np.cos(theta)
        sth = np.sin(theta)
        cps = np.cos(psi)
        sps = np.sin(psi)

        R = [[cps * cth,                    cth * sps,                         -sth],
             [cps * sph * sth - cph * sps,  cph * cps + sph * sps * sth,  cth * sph],
             [sph * sps + cph * cps * sth,  cph * sps * sth - cps * sph,  cph * cth]]

        return np.dot(R, inertial)

    def bodyToInertial(body, rotation, inertial):
        '''
         Frame-of-reference conversion routines.
         
         See Section 5 of http:#www.chrobotics.com/library/understanding-euler-angles
        '''

        phi, theta, psi = rotation

        cph = np.cos(phi)
        sph = np.sin(phi)
        cth = np.cos(theta)
        sth = np.sin(theta)
        cps = np.cos(psi)
        sps = np.sin(psi)

        R = [[cps * cth,  cps * sph * sth - cph * sps,  sph * sps + cph * cps * sth],
             [cth * sps,  cph * cps + sph * sps * sth,  cph * sps * sth - cps * sph],
             [-sth,       cth * sph,                                      cph * cth]]

        return np.dot(R, body)

    def eulerToQuaternion(eulerAngles):

        # Convenient renaming
        phi, the, psi = eulerAngles / 2

        # Pre-computation
        cph = np.cos(phi)
        cth = np.cos(the)
        cps = np.cos(psi)
        sph = np.sin(phi)
        sth = np.sin(the)
        sps = np.sin(psi)

        # Conversion
        return [[ cph * cth * cps + sph * sth * sps],
                [cph * sth * sps - sph * cth * cps],
                [-cph * sth * cps - sph * cth * sps],
                [cph * cth * sps - sph * sth * cps]]

    def __init__(self, params, motorCount):
        '''
        Constructor
        '''
        self._p = params.copy()
        self._motorCount = motorCount

        self._omegas  = np.zeros(motorCount)
        self._omegas2 = np.zeros(motorCount)

        self._x    = np.zeros(12)
        self._dxdt = np.zeros(12)

        self._airborne = False

        # Values computed in Equation 6
	self._U1 = 0     # total thrust
	self._U2 = 0     # roll thrust right
	self._U3 = 0     # pitch thrust forward
	self._U4 = 0     # yaw thrust clockwise
	self._Omega = 0  # torque clockwise

        # Exported state
        self._state = MultirotorDynamics.State()

    def computeStateDerivative(self, accelNED, netz):
        '''
        Implements Equation 12 computing temporal first derivative of state.
        Should fill _dxdx[0..11] with appropriate values.
        accelNED acceleration in NED inertial frame
        netz accelNED[2] with gravitational constant added in
        phidot rotational acceleration in roll axis
        thedot rotational acceleration in pitch axis
        psidot rotational acceleration in yaw axis
        '''
        p = self._p
        x = self._x
        Omega = self._Omega
        U2 = self._U2
        U3 = self._U3
        U4 = self._U4
 
        phidot = x[MultirotorDynamics.STATE_PHI_DOT]
        thedot = x[MultirotorDynamics.STATE_THETA_DOT]
        psidot = x[MultirotorDynamics.STATE_PSI_DOT]

        self._dxdt[0]  = x[MultirotorDynamics.STATE_X_DOT]                                                    # x'
        self._dxdt[1]  = accelNED[0]                                                                          # x''
        self._dxdt[2]  = x[MultirotorDynamics.STATE_Y_DOT]                                                    # y'
        self._dxdt[3]  = accelNED[1]                                                                          # y''
        self._dxdt[4]  = x[MultirotorDynamics.STATE_Z_DOT]                                                    # z'
        self._dxdt[5]  = netz                                                                                 # z''
        self._dxdt[6]  = phidot                                                                               # phi'
        self._dxdt[7]  = psidot * thedot * (p.Iy - p.Iz) / p.Ix - p.Jr / p.Ix * thedot * Omega + U2 / p.Ix    # phi''
        self._dxdt[8]  = thedot                                                                               # theta'
        self._dxdt[9]  = -(psidot * phidot * (p.Iz - p.Ix) / p.Iy + p.Jr / p.Iy * phidot * Omega + U3 / p.Iy) # theta''
        self._dxdt[10] = psidot                                                                               # psi'
        self._dxdt[11] = thedot * phidot * (p.Ix - p.Iy) / p.Iz + U4 / p.Iz                                   # psi''

        self._agl = 0

    def computeMotorSpeed(self, motorval):
        '''
        Computes motor speed base on motor value
        motorval motor value in [0,1]
        return motor speed in rad/s
        '''
        return motorval * self._p.maxrpm * np.pi / 30

    def init(self, rotation, airborne = False):
        '''
	Initializes kinematic pose, with flag for whether we're airbone (helps with testing gravity).
	rotation initial rotation
	airborne allows us to start on the ground (default) or in the air (e.g., gravity test)
        '''

        # Always start at location (0,0,0) with zero velocities
        self._x = np.zeros(12)

        # Support arbitrary initial rotation
        self._x[MultirotorDynamics.STATE_PHI]   = rotation[0]
        self._x[MultirotorDynamics.STATE_THETA] = rotation[1]
        self._x[MultirotorDynamics.STATE_PSI]   = rotation[2]

        # Initialize inertial frame acceleration in NED coordinates
        self._inertialAccel = MultirotorDynamics.bodyZToInertial(-MultirotorDynamics.g, rotation)

        # We usuall start on ground, but can start in air for testing
        self._airborne = airborne

    def update(self, dt):
        '''
	Updates state.
	dt time in seconds since previous update
        '''
	
        # Use the current Euler angles to rotate the orthogonal thrust vector into the inertial frame.
        # Negate to use NED.
        euler = ( self._x[6], self._x[8], self._x[10] )
        accelNED = MultirotorDynamics.bodyZToInertial(-self._U1 / self._p.m, euler)

        # We're airborne once net downward acceleration goes below zero
        netz = accelNED[2] + MultirotorDynamics.g

        #velz = self._x[MultirotorDynamics.STATE_Z_DOT]
        #debugline("Airborne: %d   AGL: %3.2f   velz: %+3.2f   netz: %+3.2f", _airborne, _agl, velz, netz)

        # If we're airborne, check for low AGL on descent
        if self._airborne:

            if self._agl <= 0 and netz >= 0:

                self._airborne = False
                self._x[MultirotorDynamics.STATE_PHI_DOT] = 0
                self._x[MultirotorDynamics.STATE_THETA_DOT] = 0
                self._x[MultirotorDynamics.STATE_PSI_DOT] = 0
                self._x[MultirotorDynamics.STATE_X_DOT] = 0
                self._x[MultirotorDynamics.STATE_Y_DOT] = 0
                self._x[MultirotorDynamics.STATE_Z_DOT] = 0

                self._x[MultirotorDynamics.STATE_PHI] = 0
                self._x[MultirotorDynamics.STATE_THETA] = 0
                self._x[MultirotorDynamics.STATE_Z] += self._agl

        # If we're not airborne, we become airborne when downward acceleration has become negative
        else:
            self._airborne = netz < 0

        # Once airborne, we can update dynamics
        if self._airborne:

            # Compute the state derivatives using Equation 12
            self.computeStateDerivative(accelNED, netz)

            # Compute state as first temporal integral of first temporal derivative
            self._x += dt * self._dxdt

            # Once airborne, inertial-frame acceleration is same as NED acceleration
            self._inertialAccel = accelNED.copy()

        else:

            #"fly" to agl=0
            vz = 5 * self._agl
            self._x[MultirotorDynamics.STATE_Z] += vz * dt

        # Get most values directly from state vector
        for i in range(3):
            ii = 2 * i
            self._state.angularVel[i]    = self._x[MultirotorDynamics.STATE_PHI_DOT + ii]
            self._state.inertialVel[i]   = self._x[MultirotorDynamics.STATE_X_DOT + ii]
            self._state.pose.rotation[i] = self._x[MultirotorDynamics.STATE_PHI + ii]
            self._state.pose.location[i] = self._x[MultirotorDynamics.STATE_X + ii]

        # Convert inertial acceleration and velocity to body frame
        self._state.bodyAccel = MultirotorDynamics.inertialToBody(self._inertialAccel, self._state.pose.rotation)

        # Convert Euler angles to quaternion
        self._state.quaternion = MultirotorDynamics.eulerToQuaternion(self._state.pose.rotation)


    def getState(self):
        '''
        Returns State class instance.
        '''
        return self._state.copy()

    def getStateVector(self):
        ''' 
        Returns "raw" state vector.
        ''' 
	return self._x.copy()

'''
private:

	# Data structure for returning state
	state_t _state = {}

	# Flag for whether we're airborne and can update dynamics
	bool _airborne = false

	# Inertial-frame acceleration
	double _inertialAccel[3] = {}

	# y = Ax + b helper for frame-of-reference conversion methods

	# Height above ground, set by kinematics
	double _agl = 0

protected:

	# state vector (see Eqn. 11) and its first temporal derivative
	double _x[12] = {}
	double _dxdt[12] = {}

	# parameter block
	Parameters* _p = NULL

	# roll right
	virtual double u2(double* o) = 0

	# pitch forward
	virtual double u3(double* o) = 0

	# yaw cw
	virtual double u4(double* o) = 0

	# radians per second for each motor, and their squared values
	double* _omegas = NULL
	double* _omegas2 = NULL

	# quad, hexa, octo, etc.
	uint8_t _motorCount = 0



public:


	/**
	 * Uses motor values to implement Equation 6.
	 *
	 * @param motorvals in interval [0,1]
	 * @param dt time constant in seconds
	 */
	virtual void setMotors(double* motorvals, double dt)
	{
		# Convert the  motor values to radians per second
		for (unsigned int i = 0 i < _motorCount ++i) {
			_omegas[i] = computeMotorSpeed(motorvals[i]) #rad/s
		}

		# Compute overall torque from omegas before squaring
		_Omega = u4(_omegas)

		# Overall thrust is sum of squared omegas
		_U1 = 0
		for (unsigned int i = 0 i < _motorCount ++i) {
			_omegas2[i] = _omegas[i] * _omegas[i]
			_U1 += _p->b * _omegas2[i]
		}

		# Use the squared Omegas to implement the rest of Eqn. 6
		_U2 = _p->l * _p->b * u2(_omegas2)
		_U3 = _p->l * _p->b * u3(_omegas2)
		_U4 = _p->d * u4(_omegas2)
	}

	/**
	 *  Gets current pose
	 *
	 *  @return data structure containing pose
	 */
	pose_t getPose(void)
	{
		pose_t pose = {}

		for (uint8_t i = 0 i < 3 ++i) {
			uint8_t ii = 2 * i
			pose.rotation[i] = _x[STATE_PHI + ii]
			pose.location[i] = _x[STATE_X + ii]
		}

		return pose
	}

	/**
	 * Sets height above ground level (AGL).
	 * This method can be called by the kinematic visualization.
	 */
	void setAgl(double agl)
	{
		_agl = agl
	}

	# Motor direction for animation
	virtual int8_t motorDirection(uint8_t i) { (void)i return 0 }

	/**
	 * Converts Euler angles to quaterion.
	 *
	 * @param eulerAngles input
	 * @param quaternion output
	 */

	/**
	 * Gets motor count set by constructor.
	 * @return motor count
	 */
	uint8_t motorCount(void)
	{
		return _motorCount
	}

} # class MultirotorDynamics
'''
