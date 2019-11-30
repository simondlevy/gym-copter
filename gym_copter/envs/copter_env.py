'''
gym-copter Environment classes

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym import Env, spaces
import numpy as np
from sys import stdout

from gym_copter.dynamics.phantom import DJIPhantomDynamics

class _CopterState:

    def __init__(self, angles=(0,0,0), altitude=0, groundspeed=0):

        self.angles = angles
        self.altitude = altitude
        self.groundspeed = groundspeed

    def __str__(self):

        return 'Pitch: %+3.1f  Roll: %+3.1f Heading: %+3.1f Altitude: %+3.1f  Groundspeed: %+3.1f' %  \
                (self.angles[0], self.angles[1], self.angles[2], self.altitude, self.groundspeed)

class _CopterEnv(Env):

    # Altitude threshold for being airborne
    ALTITUDE_MIN = 0.1

    metadata = {'render.modes': ['human']}

    def __init__(self, dt=.001):

        self.num_envs = 1
        self.dt = dt
        self.dynamics = DJIPhantomDynamics()
        self.hud = None
        self.state = _CopterState()
        self.airborne = False

    def step(self, action):

        # Update dynamics and get kinematics
        self.dynamics.setMotors(action)
        self.dynamics.update(self.dt)
        kinematics = self.dynamics.getState()

        # Get vehicle kinematics
        pose = kinematics.pose
        self.state.angles = np.degrees(pose.rotation) # HUD expects degrees
        if self.state.angles[2] < 0:
            self.state.angles[2] += 360               # Keep heading positive
        self.state.altitude = -pose.location[2]       # Location is NED, so negate Z to get altitude

        # Compute ground speed as length of X,Y velocity vector
        velocity = kinematics.inertialVel
        self.state.groundspeed = np.sqrt(velocity[0]**2 + velocity[1]**2)

        # Ascending above minimum altitude: set airborne flag
        if self.state.altitude > _CopterEnv.ALTITUDE_MIN and not self.airborne:
            self.airborne = True

        # Descending below minimum altitude: set episode-over flag
        if self.state.altitude < _CopterEnv.ALTITUDE_MIN and self.airborne:
            episode_over = True

        # Get return values 
        reward       = 0      # floating-point reward value from previous action
        episode_over = False  # whether it's time to reset the environment again (e.g., circle tipped over)
        info         = {}     # diagnostic info for debugging

        return self.state, reward, episode_over, info

    def reset(self):
        self.state = _CopterState()
        return self.state

    def render(self, mode='human'):

        from gym_copter.envs.hud import HUD

        if self.hud is None:

            self.hud = HUD()

        # Detect window close
        if not self.hud.isOpen(): return None

        return self.hud.display(mode,  self.state.angles, self.state.altitude, self.state.groundspeed) 

    def close(self):
        pass

class CopterEnvDiscreteAltitude(_CopterEnv):
    '''
    A class that rewards increased altitude.  
    Observation space (altitude) and action space (motor values) are both discretized.
    '''

    ALTITUDE_MAX = 10

    def __init__(self):

        _CopterEnv.__init__(self)

    def step(self, action):

        # Call parent-class step() to do basic update
        state, reward, episode_over, info = _CopterEnv.step(self, action)

        # Maximum altitude attained: set episode-over flag
        if state.altitude > CopterEnvAltitude.ALTITUDE_MAX:
            episode_over = True 

        reward = state.altitude

        return state, reward, episode_over, info

    def reset(self):
        _CopterEnv.reset(self)
        self.airborne = False
        return self.state

class CopterEnvRealistic(_CopterEnv):
    '''
    A class with continous state and action space.
    '''

    def __init__(self, dt=.001):

        _CopterEnv.__init__(self)

        # Action space = motors
        self.action_space = spaces.Box(np.array([0,0,0,0]), np.array([1,1,1,1]))

        # Observation space = roll,pitch,heading,altitude,groundspeed
        self.observation_space = spaces.Box(np.array([-90,-90,0,0,0]), np.array([+90,+90,359,1000,100])) 
