'''
Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym import Env, spaces
import numpy as np

from gym_copter.dynamics.phantom import DJIPhantomDynamics

class CopterEnv(Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, dt=.001):

        self.action_space = spaces.Box(np.array([0,0,0,0]), np.array([1,1,1,1]))  # motors
        self.dt = dt
        self.dynamics = DJIPhantomDynamics()
        self.hud = None
        self.angles = 0,0,0
        self.altitude = 0
        self.groundspeed = 0

    def step(self, action):

        self.dynamics.setMotors(action)
        self.dynamics.update(self.dt)

        # an environment-specific object representing your observation of the environment
        obs = self.dynamics.getState()

        reward       = 0.0   # floating-point reward value from previous action
        episode_over = False # whether it's time to reset the environment again (e.g., circle tipped over)
        info         = {}    # diagnostic info for debugging

        # Update vehicle dynamics
        self.dynamics.update(self.dt)

        # Get vehicle kinematics
        state = self.dynamics.getState()
        pose = state.pose
        self.angles = np.degrees(pose.rotation) # HUD expects degrees
        self.altitude = -pose.location[2]       # Location is NED, so negate Z to get altitude

        # Compute ground speed as length of X,Y velocity vector
        velocity = state.inertialVel
        groundspeed = np.sqrt(velocity[0]**2 + velocity[1]**2)

        return obs, reward, episode_over, info

    def reset(self):
        pass

    def render(self, mode='human'):

        from gym_copter.envs.hud import HUD

        if self.hud is None:

            self.hud = HUD()

        # Detect window close
        if not self.hud.isOpen(): return None

        return self.hud.display(mode,  self.angles, self.altitude, self.groundspeed) 

    def close(self):
        pass

