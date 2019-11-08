'''
Copyright (C) 2019 Simon D. Levy

MIT License
'''

import gym
from gym import spaces
import numpy as np

import pyglet

from gym_copter.dynamics.phantom import DJIPhantomDynamics

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
        self.dynamics = DJIPhantomDynamics()
        self.viewer = None
        self.heading_widgets = []

        # XXX Mock up heading for now
        self.heading = 0

    def step(self, action):

        self.dynamics.setMotors(action)
        self.dynamics.update(self.dt)

        # an environment-specific object representing your observation of the environment
        obs = self.dynamics.getState()

        reward       = 0.0   # floating-point reward value from previous action
        episode_over = False # whether it's time to reset the environment again (e.g., circle tipped over)
        info         = {}    # diagnostic info for debugging

        self.dynamics.update(self.dt)

        return obs, reward, episode_over, info

    def reset(self):
        pass

    def render(self, mode='human'):

        # Adapted from https://raw.githubusercontent.com/openai/gym/master/gym/envs/classic_control/cartcircle.py

        from gym.envs.classic_control import rendering

        W = 800
        H = 500

        self.w = W
        self.h = H

        if self.viewer is None:

            self.viewer = rendering.Viewer(W, H)

            sky = rendering.FilledPolygon([(0,H), (0,0), (W,0), (W,H)])
            sky.set_color(0.5,0.8,1.0)
            self.viewer.add_geom(sky)

            self.altitude_label = pyglet.text.Label('0000', font_size=24, x=600, y=300, 
                    anchor_x='left', anchor_y='center', color=(0,0,0,255))
            self.viewer.add_geom(_DrawText(self.altitude_label))

            # Add a horizontal line at the top for the yaw display
            dy = 35
            self.viewer.add_geom(self.viewer.draw_line((0,H-dy), (W,H-dy), color=(1.0,1.0,1.0)))

            # Ground will be replaced on each call to render()
            self.ground = None

        # Detect window close
        if not self.viewer.isopen: return None

        # Get vehicle state
        state = self.dynamics.getState()
        pose = state.pose
        location = pose.location
        rotation = pose.rotation

        # We're using NED frame, so negate altitude before displaying
        self.altitude_label.text = "Alt: %5.2fm" % -location[2]

        self.altitude_label.draw()

        # Remove previoud ground quadrilateral
        if not self.ground is None:
            del self.ground

        # Center top of ground quadrilateral depends on pitch
        y = H/2 * (1 + np.sin(rotation[1]))

        # Left and right top of ground quadrilateral depend on roll
        dy = W/2 * np.sin(rotation[0])
        ury = y + dy
        uly = y - dy

        # Draw new ground quadrilateral:         LL     LR     UR     UL
        self.ground = self.viewer.draw_polygon([(0,0), (W,0), (W,ury), (0,uly),], color=(0.5, 0.7, 0.3) )

        # Display heading
        self._show_heading(rotation[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):

        pass

 
    def _show_heading(self, heading):

        for item in self.heading_widgets:
            del item

        # Round heading to nearest 15 degrees
        heading = (self.heading // 15) * 15

        tickvals = range(heading-60, heading+75, 15)

        print(heading, list(tickvals))
        stdout.flush()

        self.heading = (self.heading+1) % 360

        #line = self.viewer.draw_line((self.w/2,self.h-dy), (self.w/2,self.h-dy/2), color=(1.0,1.0,1.0))
        #self.viewer.add_geom(line)

        #r = 200
        #a0 = 45
        #dy = 250 
        #points = [(w/2+np.cos(np.radians(a))*r, h-dy+np.sin(np.radians(a))*r) for a in range(a0,180-a0+1)]
        #self.viewer.draw_polyline(points, color=(1.0,1.0,1.0), linewidth=2)
