'''
Copyright (C) 2019 Simon D. Levy

MIT License
'''

import gym
from gym import spaces
import numpy as np

import pyglet

from gym_copter.dynamics.phantom import DJIPhantomDynamics

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
        W                       = 800 # window width
        H                       = 500 # window height
        SKY_COLOR               = 0.5, 0.8, 1.0
        GROUND_COLOR            = 0.5, 0.7, 0.3
        LINE_COLOR              = 1.0, 1.0, 1.0
        HEADING_TICK_SPACING    = 80
        HEADING_TICK_COUNT      = 24
        HEADING_LABEL_Y_OFFSET  = 17
        HEADING_LINE_Y_OFFSET   = 35
        FONT_SIZE               = 18
        FONT_COLOR              = 255,255,255
        PITCH_LINE_SPACING      = 40
        PITCH_LINE_WIDTH        = 30
        PITCH_LABEL_X_OFFSET    = 40
        PITCH_LABEL_Y_OFFSET    = 0
        POINTER_COLOR           = 1.0, 0.0, 0.0
        HEADING_POINTER_SIZE    = 5
        ALTITUDE_BOX_HEIGHT     = 200
        ALTITUDE_BOX_WIDTH      = 90
        ALTITUDE_BOX_X_MARGIN   = 10
        ALTITUDE_LABEL_OFFSET   = 60
        ALTITUDE_POINTER_SIZE   = 8
        ALTITUDE_STEP_METERS    = 5
        ALTITUDE_STEP_PIXELS    = 8
        ROLL_RETICLE_RADIUS     = 225
        ROLL_RETICLE_LIM        = 45
        ROLL_RETICLE_PTS        = 100
        ROLL_RETICLE_YOFF       = 200
        ROLL_RETICLE_STRIDE     = 10
        ROLL_RETICLE_TICKLEN    = 10
        ROLL_RETICLE_TICKVALS   = [10, 20, 30, 45, 60]
 
        from gym.envs.classic_control import rendering
        from pyglet.gl import glTranslatef, glLoadIdentity, glRotatef

        # https://stackoverflow.com/questions/56744840/pyglet-label-not-showing-on-screen-on-draw-with-openai-gym-render

        class _DrawText:
            def __init__(self, label:pyglet.text.Label):
                self.label=label
            def render(self):
                self.label.draw()

        class _DrawTextRotated:
            def __init__(self, label:pyglet.text.Label, x, y, phi):
                self.label=label
                self.x = x
                self.y = y
                self.phi = phi
            def render(self):
                glTranslatef(self.x, self.y, 0)
                glRotatef(np.degrees(self.phi), 0.0, 0.0, 1.0)
                self.label.draw()
                glLoadIdentity() # Restores ordinary drawing

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
                y=H-HEADING_LABEL_Y_OFFSET, color=(*FONT_COLOR,255), 
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

        phi = rotation[0]

        # Left and right top of ground quadrilateral depend on roll
        dx,dy = _rotate(W, 0, phi)
        x1 = cx - dx
        y1 = gcy - dy
        x2 = cx + dx
        y2 = gcy + dy

        # Draw new ground quadrilateral         
        self.viewer.draw_polygon([(x1,y1), (x2,y2), (x2,y2-2*H), (x1,y1-2*H)], color=GROUND_COLOR)

        # Add a reticle for pitch, rotated by roll to match horizon
        for i in range(-3,4):

            x1 = 0
            y1 = i * PITCH_LINE_SPACING

            x2 = x1 + PITCH_LINE_WIDTH + (1-(i%2))*PITCH_LINE_WIDTH/2 # alternate line length
            y2 = y1

            x1r,y1r = _rotate(x1, y1, phi)
            x2r,y2r = _rotate(x2, y2, phi)

            # Draw two sets of lines for thickness
            self.viewer.draw_line((cx+x1r,cy+y1r),   (cx+x2r,cy+y2r), color=LINE_COLOR)
            self.viewer.draw_line((cx+x1r,cy+y1r+1), (cx+x2r,cy+y2r+1), color=LINE_COLOR)
            self.viewer.draw_line((cx-x1r,cy-y1r),   (cx-x2r,cy-y2r), color=LINE_COLOR)
            self.viewer.draw_line((cx-x1r,cy-y1r+1), (cx-x2r,cy-y2r+1), color=LINE_COLOR)

            # Add a label on the left of every other tick
            if i%2 == 0:
                pitch_label = pyglet.text.Label(('%+3d'%(-i*10)).center(3), 
                        font_size=FONT_SIZE, color=(*FONT_COLOR,255), 
                        anchor_x='center', anchor_y='center') 
                label_x = cx-x2r-PITCH_LABEL_X_OFFSET 
                label_y = cy-y2r-PITCH_LABEL_Y_OFFSET
                self.viewer.add_onetime(_DrawTextRotated(pitch_label, label_x, label_y, phi))

        # Add a horizontal line and triangular pointer at the top for the heading display
        self.viewer.draw_line((0,H-HEADING_LINE_Y_OFFSET), (W,H-HEADING_LINE_Y_OFFSET), color=LINE_COLOR)
        self.viewer.draw_polygon([
            (W/2-HEADING_POINTER_SIZE,H-HEADING_LINE_Y_OFFSET-HEADING_POINTER_SIZE), 
            (W/2+HEADING_POINTER_SIZE,H-HEADING_LINE_Y_OFFSET-HEADING_POINTER_SIZE), 
            (W/2,H-HEADING_LINE_Y_OFFSET+HEADING_POINTER_SIZE)],
            color=POINTER_COLOR)

        # Display heading
        for i,heading_label in enumerate(self.heading_labels):
            d = HEADING_TICK_SPACING * HEADING_TICK_COUNT
            x = (W/2 - heading*d/360 + HEADING_TICK_SPACING*i) % d
            self.viewer.add_onetime(_DrawText(heading_label))
            heading_label.x = x

        # Add a box and a pointer on the right side for the altitude gauge
        l = W - ALTITUDE_BOX_WIDTH - ALTITUDE_BOX_X_MARGIN
        r = W - ALTITUDE_BOX_X_MARGIN
        b = H/2 - ALTITUDE_BOX_HEIGHT/2
        t = H/2 + ALTITUDE_BOX_HEIGHT/2
        self.viewer.draw_polygon([(l,t),(r,t),(r,b),(l,b)], color=LINE_COLOR, linewidth=2, filled=False)
        self.viewer.draw_polygon([
            (l,H/2-ALTITUDE_POINTER_SIZE), (l,H/2+ALTITUDE_POINTER_SIZE), (l+ALTITUDE_POINTER_SIZE,H/2)], 
            color=POINTER_COLOR)

        # Display altitude in the box
        closest = altitude // ALTITUDE_STEP_METERS * ALTITUDE_STEP_METERS
        for k in range(-2,3):
            tickval = closest+k*ALTITUDE_STEP_METERS
            diff = tickval - altitude
            dy = diff*ALTITUDE_STEP_PIXELS

            # Avoid putting tick label below bottom of box
            if dy > -ALTITUDE_BOX_HEIGHT/2:

                # Use a non-linear fade-in/out for numbers at top, bottom
                alpha = int(255  * np.sqrt(max(0, (1-abs(diff)/10.))))
                altitude_label = pyglet.text.Label(('%3d'%tickval).center(3), x=W-ALTITUDE_LABEL_OFFSET, y=H/2+dy,
                        font_size=FONT_SIZE, color=(*FONT_COLOR,alpha), anchor_x='center', anchor_y='center') 
                self.viewer.add_onetime(_DrawText(altitude_label))

        # Add a reticle at the top for roll
        angles = np.linspace(np.radians(180-ROLL_RETICLE_LIM), np.radians(ROLL_RETICLE_LIM), ROLL_RETICLE_PTS)
        points = [(np.cos(a)*ROLL_RETICLE_RADIUS+W/2, np.sin(a)*ROLL_RETICLE_RADIUS+ROLL_RETICLE_YOFF) for a in angles]
        self.viewer.draw_polyline(points, color=LINE_COLOR, linewidth=2)
        tickvals = np.append(-np.array(ROLL_RETICLE_TICKVALS[::-1]), [0] + ROLL_RETICLE_TICKVALS)
        for tickval in tickvals: 
            k = int((ROLL_RETICLE_PTS-1) * (tickval-tickvals[0]) / (tickvals[-1]-tickvals[0]))
            x1,y1 = points[k]
            x2,y2 = x1,y1+ROLL_RETICLE_TICKLEN
            xr,yr = _rotate(0, ROLL_RETICLE_TICKLEN, np.radians(-ROLL_RETICLE_TICKVALS[-1]/ROLL_RETICLE_LIM*tickval))
            self.viewer.draw_line((x1,y1),  (x2+xr, y2+yr), color=LINE_COLOR)
            #roll_label = pyglet.text.Label(('%3d'%0).center(3), x=x2, y=y2+10,
            #        font_size=FONT_SIZE, color=(*FONT_COLOR,255), anchor_x='center', anchor_y='center') 
            #self.viewer.add_onetime(_DrawText(roll_label))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        pass

