'''
gym-copter Environment classes

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym import Env, spaces
import numpy as np
from sys import stdout

from gym_copter.dynamics.phantom import DJIPhantomDynamics

class _CopterEnv(Env):

    # Altitude threshold for being airborne
    ALTITUDE_MIN = 0.1

    metadata = {'render.modes': ['human']}

    def __init__(self, dt=.001):

        self.num_envs = 1
        self.dt = dt
        self.dynamics = DJIPhantomDynamics()
        self.hud = None
        self.state = np.zeros(12)
        self.altitude = 0
        self.airborne = False

    def step(self, action):

        # Update dynamics and get kinematic state
        self.dynamics.setMotors(action)
        self.dynamics.update(self.dt)
        self.state = self.dynamics.getState()

        # Special handling for altitude: negate to accommodate NED; then save it
        self.altitude = -self.state[4]

        # Ascending above minimum altitude: set airborne flag
        if self.altitude > _CopterEnv.ALTITUDE_MIN and not self.airborne:
            self.airborne = True

        # Descending below minimum altitude: set episode-over flag
        if self.altitude < _CopterEnv.ALTITUDE_MIN and self.airborne:
            episode_over = True

        # Get return values 
        reward       = 0      # floating-point reward value from previous action
        episode_over = False  # whether it's time to reset the environment again (e.g., circle tipped over)
        info         = {}     # diagnostic info for debugging

        return self.state, reward, episode_over, info

    def reset(self):
        self.state = np.zeros(12)
        return self.state

    def render(self, mode='human'):

        from gym_copter.envs.hud import HUD

        if self.hud is None:

            self.hud = HUD()

        # Detect window close
        if not self.hud.isOpen(): return None

        return self.hud.display(mode,  self.state)

    def close(self):
        pass

class CopterEnvAltitudeRewardDiscreteMotors(_CopterEnv):
    '''
    A class that rewards increased altitude.  
    Action space (motor values) is discretized.
    '''

    ALTITUDE_MAX = 10
    MOTOR_STEPS  = 5

    def __init__(self):

        _CopterEnv.__init__(self)

        # Action space = motors, discretize to intervals
        self.action_space = spaces.Discrete((CopterEnvAltitudeRewardDiscreteMotors.MOTOR_STEPS+1)**4)

        # Observation space = altitude
        self.observation_space = spaces.Box(np.array([0]), np.array([np.inf]))

    def step(self, action):

        print(action, (((action//1)%6)/5., ((action//6)%6)/5., ((action//36)%6)/5., ((action//216)%6)/5.))
        exit(0)

        # Call parent-class step() to do basic update
        state, reward, episode_over, info = _CopterEnv.step(self, action)

        # Maximum altitude attained: set episode-over flag
        if self.altitude > CopterEnvAltitude.ALTITUDE_MAX:
            episode_over = True 

        # Altitude is both the state and the reward
        return np.array([self.altitude]), self.altitude, episode_over, info

    def reset(self):
        _CopterEnv.reset(self)
        self.airborne = False
        return np.array([self.altitude])

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
