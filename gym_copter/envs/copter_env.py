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

        # Screen size, pixels
        W = 800
        H = 500

        # Altitude
        ALTITUDE_SPAN_METERS    = 200
        ALTITUDE_STEP_METERS    = 5
        ALTITUDE_SPACING_PIXELS = 40

        self.w = W
        self.h = H

        self.heading_spacing = 80

        if self.viewer is None:

            self.viewer = rendering.Viewer(W, H)

            # Add sky as backround
            sky = rendering.FilledPolygon([(0,H), (0,0), (W,0), (W,H)])
            sky.set_color(0.5,0.8,1.0)
            self.viewer.add_geom(sky)

            # Create labels for heading
            self.heading_labels = [pyglet.text.Label(('%d'%(c*15)).center(3), font_size=20, y=H-17, 
                color=(255,255,255,255), anchor_x='center', anchor_y='center') for c in range(24)]

        # Detect window close
        if not self.viewer.isopen: return None

        # Get vehicle state
        state = self.dynamics.getState()
        pose = state.pose
        location = pose.location
        rotation = pose.rotation
        altitude = -location[2]
        heading  = np.degrees(rotation[2])

        # Center top of ground quadrilateral depends on pitch
        cy = H/2 * (1 + np.sin(rotation[1]))

        # XXX Fix roll at 45 deg for testing
        #phi = rotation[0]
        phi = np.pi / 8 

        # Left and right top of ground quadrilateral depend on roll
        dx = np.cos(phi)*W
        dy = np.sin(phi)*W
        cx = W / 2
        x1 = cx - dx
        y1 = cy - dy
        x2 = cx + dx
        y2 = cy + dy

        # Draw new ground quadrilateral         
        self.viewer.draw_polygon([(x1,y1), (x2,y2), (x2,y2-2*H), (x1,y1-2*H)], color=(0.5, 0.7, 0.3) )

        # Add a reticule for pitch, rotated by roll to match horizon
        #for k in range(-4,5):
        for k in range(0,1):
            dx = np.cos(phi)*W
            dy = np.sin(phi)*W
            cx = W / 2
            x1 = cx - dx
            y1 = cy - dy
            x2 = cx + dx
            y2 = cy + dy
            self.viewer.draw_line((x1,y1), (x2,y2), color=(1.0,1.0,1.0))

        # Add a horizontal line and pointer at the top for the heading display
        self.viewer.draw_line((0,H-35), (W,H-35), color=(1.0,1.0,1.0))
        self.viewer.draw_polygon([(self.w/2-5,self.h-40), (self.w/2+5,self.h-40), (400,self.h-30)], color=(1.0,0.0,0.0))

        # Display heading
        for i,heading_label in enumerate(self.heading_labels):
            x = (self.w/2 - heading * 5.333333 + self.heading_spacing*i) % 1920
            self.viewer.add_onetime(_DrawText(heading_label))
            heading_label.x = x

        # Add a box and a pointer on the right side for the altitude gauge
        h2 = 100
        l = self.w - 100
        r = self.w - 10
        b = self.h/2 - h2
        t = self.h/2 + h2
        self.viewer.draw_polygon([(l,t),(r,t),(r,b),(l,b)], color=(1.0, 1.0, 1.0), linewidth=2, filled=False)
        self.viewer.draw_polygon([(l,self.h/2-8), (l,self.h/2+8), (l+8,self.h/2)], color=(1.0,0.0,0.0))

        # Display altitude in the box
        closest = altitude // ALTITUDE_STEP_METERS * ALTITUDE_STEP_METERS
        for k in range(-2,3):
            tickval = closest+k*ALTITUDE_STEP_METERS
            diff = tickval - altitude
            dy = 8*diff

            # Avoid putting tick label below bottom of box
            if dy > -100:

                # Use a non-linear fade-in/out for numbers at top, bottom
                alpha = int(255  * np.sqrt(max(0, (1-abs(diff)/10.))))
                altitude_label = pyglet.text.Label(('%3d'%tickval).center(3), x=W-60, y=self.h/2+dy,
                        font_size=20, color=(255,255,255,alpha), anchor_x='center', anchor_y='center') 
                self.viewer.add_onetime(_DrawText(altitude_label))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):

        pass
