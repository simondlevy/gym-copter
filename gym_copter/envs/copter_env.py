'''
Copyright (C) 2019 Simon D. Levy

MIT License
'''

import gym
from gym import spaces
import numpy as np

import pyglet

from gym_copter.dynamics.phantom import DJIPhantomDynamics

# https://stackoverflow.com/questions/56744840/pyglet-label-not-showing-on-screen-on-draw-with-openai-gym-render
class _DrawText:
    def __init__(self, label:pyglet.text.Label):
        self.label=label
    def render(self):
        self.label.draw()

class CopterEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, dt=.001):

        self.action_space = spaces.Box(np.array([0,0,0,0]), np.array([1,1,1,1]))  # motors
        self.dt = dt
        self.dynamics = DJIPhantomDynamics()
        self.viewer = None

    def step(self, action):

        self.dynamics.setMotors(action)
        self.dynamics.update(self.dt)

        # an environment-specific object representing your observation of the environment
        obs = self.dynamics.getState()

        reward       = 0.0   # floating-point reward value from previous action
        episode_over = False # whether it's time to reset the environment again (e.g., pole tipped over)
        info         = {}    # diagnostic info for debugging

        self.dynamics.update(self.dt)

        return obs, reward, episode_over, info

    def reset(self):
        pass

    def render(self, mode='human'):

        # Adapted from https://raw.githubusercontent.com/openai/gym/master/gym/envs/classic_control/cartpole.py

        SCREEN_WIDTH = 800
        SCREEN_HEIGHT = 500

        if self.viewer is None:

            from gym.envs.classic_control import rendering

            def rect(r, b, l=0, t=0):
                return rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])

            self.viewer = rendering.Viewer(SCREEN_WIDTH, SCREEN_HEIGHT)
            sky = rect(SCREEN_WIDTH, SCREEN_HEIGHT)
            sky.set_color(0.5,0.8,1.0)

            self.viewer.add_geom(sky)
            self.altitude_label = pyglet.text.Label('0000', font_size=24, x=600, y=300, 
                    anchor_x='left', anchor_y='center', color=(0,0,0,255))
            self.viewer.add_geom(_DrawText(self.altitude_label))

            self.foo = 0
            self.ground = None

        # Detect window close
        if not self.viewer.isopen: return None

        state = self.dynamics.getState()
        pose = state.pose
        location = pose.location
        rotation = pose.rotation

        # We're using NED frame, so negate altitude before displaying
        self.altitude_label.text = "Alt: %5.2fm" % -location[2]

        self.foo += 1
        offset = int(50 * np.sin(self.foo))

        self.altitude_label.draw()

        if not self.ground is None:
            del self.ground

        self.ground = self.viewer.draw_polygon( [
            (self.foo,                  0),
            (self.foo+600/30.0, 0),
            (self.foo+600/30.0, 400/30.0),
            (self.foo,                  400/30.0),
            ], color=(0.5, 0.7, 0.3) )

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        pass
