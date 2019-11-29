'''
gym-copter Environment classes

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym import Env, spaces
import numpy as np

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

    metadata = {'render.modes': ['human']}

    def __init__(self, dt=.001):

        self.num_envs = 1
        self.action_space = spaces.Box(np.array([0,0,0,0]), np.array([1,1,1,1]))  # motors
        self.observation_space = spaces.Box(np.array([-90,-90,0,0,0]), np.array([+90,+90,359,1000,100]))  # motors
        self.dt = dt
        self.dynamics = DJIPhantomDynamics()
        self.hud = None
        self.state = _CopterState()

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

        # Get return values 
        reward       = self._getReward()  # floating-point reward value from previous action
        episode_over = False              # whether it's time to reset the environment again (e.g., circle tipped over)
        info         = {}                 # diagnostic info for debugging

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

class CopterEnvAltitude(_CopterEnv):
    '''
    A class that rewards increased altitude
    '''

    def step(self, action):

        state, reward, _, info = _CopterEnv.step(self, action)

        # Episode is over when copter reaches 100m altitude or returns to earth
        episode_over = False 

        return state, reward, episode_over, info

    def _getReward(self):

        return self.state.altitude
