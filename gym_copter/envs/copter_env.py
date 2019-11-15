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

        # Arbitrary constants
        W                     = 800 # window width
        H                     = 500 # window height
        ALTITUDE_STEP_METERS  = 5
        SKY_COLOR             = 0.5, 0.8, 1.0
        GROUND_COLOR          = 0.5, 0.7, 0.3
        LINE_COLOR            = 1.0, 1.0, 1.0
        HEADING_SPACING       = 80
        HEADING_TICK_COUNT    = 24
        HEADING_TICK_Y_OFFSET = 17
        FONT_SIZE             = 20
        FONT_COLOR            = 255,255,255
        PITCH_LINE_SPACING    = 20
        PITCH_LINE_WIDTH      = 30

        from gym.envs.classic_control import rendering

        # https://stackoverflow.com/questions/56744840/pyglet-label-not-showing-on-screen-on-draw-with-openai-gym-render
        class _DrawText:
            def __init__(self, label:pyglet.text.Label):
                self.label=label
            def render(self):
                self.label.draw()

        def _rotate(x, y, phi):
            return np.cos(phi)*x - np.sin(phi)*y, np.sin(phi)*x + np.cos(phi)*y

        if self.viewer is None:

            self.viewer = rendering.Viewer(W, H)

            # Add sky as backround
            sky = rendering.FilledPolygon([(0,H), (0,0), (W,0), (W,H)])
            sky.set_color(*SKY_COLOR)
            self.viewer.add_geom(sky)

            # Create labels for heading
            self.heading_labels = [pyglet.text.Label(('%d'%(c*360//HEADING_TICK_COUNT)).center(3), font_size=FONT_SIZE, 
                y=H-HEADING_TICK_Y_OFFSET, color=(*FONT_COLOR,255), 
                anchor_x='center', anchor_y='center') for c in range(HEADING_TICK_COUNT)]

        # Detect window close
        if not self.viewer.isopen: return None

        # Get vehicle state
        state = self.dynamics.getState()
        pose = state.pose
        location = pose.location
        rotation = pose.rotation
        altitude = -location[2]
        heading  = np.degrees(rotation[2])

        # Get center coordinates
        cx,cy = W/2, H/2

        # Center vertical of ground depends on pitch
        gcy = H/2 * (1 + np.sin(rotation[1]))

        # XXX Fix roll at 45 deg for testing
        phi = rotation[0]
        #phi = np.pi / 8 

        # Left and right top of ground quadrilateral depend on roll
        dx,dy = _rotate(W, 0, phi)
        x1 = cx - dx
        y1 = gcy - dy
        x2 = cx + dx
        y2 = gcy + dy

        # Draw new ground quadrilateral         
        self.viewer.draw_polygon([(x1,y1), (x2,y2), (x2,y2-2*H), (x1,y1-2*H)], 
                color=(GROUND_COLOR[0], GROUND_COLOR[1], GROUND_COLOR[2]))

        # Add a reticule for pitch, rotated by roll to match horizon
        for i in range(-3,4):

            x1 = 0
            y1 = i * PITCH_LINE_SPACING

            x2 = x1 + PITCH_LINE_WIDTH + (1-(i%2))*PITCH_LINE_WIDTH/2 # alternate line length
            y2 = y1

            x1r,y1r = _rotate(x1, y1, phi)
            x2r,y2r = _rotate(x2, y2, phi)

            # Draw two sets of lines for thickness
            for j in (0,1):
                for k in (-1,+1):
                    self.viewer.draw_line((cx+k*x1r,cy+k*y1r+j), (cx+k*x2r,cy+k*y2r+j), 
                            color=(LINE_COLOR[0], LINE_COLOR[1], LINE_COLOR[2]))
 
            pitch_label = pyglet.text.Label(('%+3d'%(i*5)).center(3), x=cx-x1r, y=cy-y1r,
                        font_size=20, color=(*FONT_COLOR,255), anchor_x='center', anchor_y='center') 
            self.viewer.add_onetime(_DrawText(pitch_label))

            

        # Add a horizontal line and triangular pointer at the top for the heading display
        self.viewer.draw_line((0,H-35), (W,H-35), color=(LINE_COLOR[0], LINE_COLOR[1], LINE_COLOR[2]))
        self.viewer.draw_polygon([(W/2-5,H-40), (W/2+5,H-40), (400,H-30)], color=(1.0,0.0,0.0))

        # Display heading
        for i,heading_label in enumerate(self.heading_labels):
            x = (W/2 - heading * 5.333333 + HEADING_SPACING*i) % 1920
            self.viewer.add_onetime(_DrawText(heading_label))
            heading_label.x = x

        # Add a box and a pointer on the right side for the altitude gauge
        h2 = 100
        l = W - 100
        r = W - 10
        b = H/2 - h2
        t = H/2 + h2
        self.viewer.draw_polygon([(l,t),(r,t),(r,b),(l,b)], color=(1.0, 1.0, 1.0), linewidth=2, filled=False)
        self.viewer.draw_polygon([(l,H/2-8), (l,H/2+8), (l+8,H/2)], color=(1.0,0.0,0.0))

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
                altitude_label = pyglet.text.Label(('%3d'%tickval).center(3), x=W-60, y=H/2+dy,
                        font_size=20, color=(*FONT_COLOR,alpha), anchor_x='center', anchor_y='center') 
                self.viewer.add_onetime(_DrawText(altitude_label))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        pass
