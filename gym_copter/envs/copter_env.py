'''
Copyright (C) 2019 Simon D. Levy

MIT License
'''

import gym
from gym import spaces
import numpy as np

import pyglet

from gym_copter.dynamics.inspire import DJIInspireDynamics

from sys import stdout

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
        self.dynamics = DJIInspireDynamics()
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

        # Adapted from https://raw.githubusercontent.com/openai/gym/master/gym/envs/classic_control/groundpole.py

        screen_width = 800
        screen_height = 500

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = 0, screen_width, 0, screen_height
            sky = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            sky.set_color(0.5,0.8,1.0)
            t -= screen_height / 2
            b -= screen_height / 2
            ground = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            ground.set_color(0.5, 0.7 , 0.3)
            self.groundtrans = rendering.Transform()
            ground.add_attr(self.groundtrans)
            self.viewer.add_geom(sky)
            self.viewer.add_geom(ground)
            self.altitude_label = pyglet.text.Label('0000', font_size=24, x=600, y=300, 
                    anchor_x='left', anchor_y='center', color=(0,0,0,255))
            self.viewer.add_geom(_DrawText(self.altitude_label))

            self.foo = 0

        # Detect window close
        if not self.viewer.isopen: return None

        state = self.dynamics.getState()
        pose = state.pose
        location = pose.location
        rotation = pose.rotation

        # We're using NED frame, so negate altitude before displaying
        self.altitude_label.text = "Alt: %5.2fm" % -location[2]

        self.foo += .01
        offset = int(50 * np.sin(self.foo))
        stdout.flush()

        self.groundtrans.set_translation(0, offset)

        self.altitude_label.draw()

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        pass
