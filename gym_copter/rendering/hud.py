'''
Heads-Up Display using gym.envs.classic_control.rendering / pyglet

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import numpy as np
from gym.envs.classic_control import rendering
from pyglet.gl import glTranslatef, glLoadIdentity, glRotatef
from pyglet.text import Label


class _DrawText:
    '''
    https://stackoverflow.com/questions/56744840
    '''

    def __init__(self, label):
        self.label = label

    def render(self):
        self.label.draw()


class _DrawTextRotated:

    def __init__(self, label, x, y, angle, xoff=0):
        self.label = label
        self.x = x
        self.y = y
        self.angle = angle
        self.xoff = xoff

    def render(self):
        glTranslatef(self.x, self.y, 0)
        glRotatef(self.angle, 0.0, 0.0, 1.0)
        glTranslatef(self.xoff, 0, 0)
        self.label.draw()
        glLoadIdentity()  # Restores ordinary drawing


class HUD:

    # Arbitrary constants
    W = 800  # window width
    H = 600  # window height
    SKY_COLOR = 0.5, 0.8, 1.0
    GROUND_COLOR = 0.5, 0.7, 0.3
    LINE_COLOR = 1.0, 1.0, 1.0
    HIGHLIGHT_COLOR = 0.5, 0.5, 0.5
    POINTER_COLOR = 1.0, 0.0, 0.0
    HEADING_TICK_SPACING = 80
    HEADING_TICK_COUNT = 24
    HEADING_LABEL_Y_OFFSET = 17
    HEADING_LINE_Y_OFFSET = 35
    HEADING_BOX_WIDTH = 20
    FONT_SIZE = 18
    SMALL_FONT_SIZE = 14
    LARGE_FONT_SIZE = 20
    FONT_COLOR = 255, 255, 255
    PITCH_RETICLE_SPACING = 40
    PITCH_RETICLE_INCREMENT = 10
    PITCH_RETICLE_WIDTH = 30
    PITCH_LABEL_X_OFFSET = 40
    PITCH_LABEL_Y_OFFSET = 0
    VERTICAL_BOX_HEIGHT = 300
    VERTICAL_BOX_WIDTH = 90
    VERTICAL_LABEL_OFFSET = 30
    VERTICAL_POINTER_HEIGHT = 15
    VERTICAL_STEP_METERS = 5
    VERTICAL_STEP_PIXELS = 8
    VERTICAL_TITLE_X_OFFSET = 45
    VERTICAL_TITLE_Y_OFFSET = 20
    ROLL_RETICLE_RADIUS = 300
    ROLL_RETICLE_LIM = 45
    ROLL_RETICLE_PTS = 100
    ROLL_RETICLE_YOFF = 200
    ROLL_RETICLE_STRIDE = 10
    ROLL_RETICLE_TICKLEN = 5
    ROLL_RETICLE_TICK_YOFF = 25
    ROLL_RETICLE_TICKVALS = [10, 20, 30, 45, 60]
    ROLL_POINTER_SIZE = 10
    TIME_LABEL_X = 400
    TIME_LABEL_Y = 50

    def _rotate(x, y, angle):
        angle = np.radians(angle)
        return (np.cos(angle)*x - np.sin(angle)*y,
                np.sin(angle)*x + np.cos(angle)*y)

    def _tickval2index(tickval, tickvals):
        return int((HUD.ROLL_RETICLE_PTS-1) *
                   (tickval-tickvals[0]) / (tickvals[-1]-tickvals[0]))

    def _add_label(viewer, label):
        viewer.add_onetime(_DrawText(label))

    def _add_label_rotated(viewer, label, x, y, angle, xoff=0):
        viewer.add_onetime(_DrawTextRotated(label, x, y, angle, xoff))

    def _vertical_display(viewer, leftx, stripx, value, title):

        dy = HUD.VERTICAL_POINTER_HEIGHT

        # Display a tapered strip in the middle for highlighting current value
        stripw = HUD.VERTICAL_BOX_WIDTH + dy
        x1, y1 = stripx, 0+HUD.H/2
        x2, y2 = stripx+dy, dy+HUD.H/2
        x3, y3 = stripx+stripw-dy, dy+HUD.H/2
        x4, y4 = stripx+stripw, 0+HUD.H/2
        x5, y5 = stripx+stripw-dy, -dy+HUD.H/2
        x6, y6 = stripx+dy, -dy+HUD.H/2
        viewer.draw_polygon([(x1, y1),
                             (x2, y2),
                             (x3, y3),
                             (x4, y4),
                             (x5, y5),
                             (x6, y6)],
                            color=HUD.HIGHLIGHT_COLOR)

        # Display a box for the gauge
        lx = leftx
        rx = lx + HUD.VERTICAL_BOX_WIDTH
        b = HUD.H/2 - HUD.VERTICAL_BOX_HEIGHT/2
        t = HUD.H/2 + HUD.VERTICAL_BOX_HEIGHT/2
        viewer.draw_polygon([(lx, t), (rx, t), (rx, b), (lx, b)],
                            color=HUD.LINE_COLOR, linewidth=2, filled=False)

        # Display the current values in the box
        closest = value // HUD.VERTICAL_STEP_METERS * HUD.VERTICAL_STEP_METERS
        for k in range(-3, 4):
            tickval = closest+k*HUD.VERTICAL_STEP_METERS
            diff = tickval - value
            dy = diff*HUD.VERTICAL_STEP_PIXELS

            # Use a linear fade-in/out for numbers at top, bottom
            alpha = int(255 * (HUD.VERTICAL_BOX_HEIGHT/2 - abs(dy)) /
                        (HUD.VERTICAL_BOX_HEIGHT/2.))

            # Avoid putting tick label below bottom of box
            if dy > -HUD.VERTICAL_BOX_HEIGHT/2+20:
                label = Label(('%3d' % tickval).center(3),
                              x=lx+HUD.VERTICAL_LABEL_OFFSET,
                              y=HUD.H/2+dy,
                              font_size=HUD.FONT_SIZE,
                              color=(*HUD.FONT_COLOR, alpha),
                              anchor_x='center',
                              anchor_y='center')
                viewer.add_onetime(_DrawText(label))

        # Add a title at the bottom
        HUD._add_label(viewer,
                       Label(title,
                             x=lx+HUD.VERTICAL_TITLE_X_OFFSET,
                             y=(HUD.H/2-HUD.VERTICAL_BOX_HEIGHT/2 -
                                HUD.VERTICAL_TITLE_Y_OFFSET),
                             font_size=HUD.FONT_SIZE,
                             color=(*HUD.FONT_COLOR, 255),
                             anchor_x='center', anchor_y='center'))

    def __init__(self, env):

        env = env.unwrapped
        self.env = env
        self.env.viewer = self

        self.viewer = rendering.Viewer(HUD.W, HUD.H)

        # Add sky as backround
        sky = rendering.FilledPolygon([(0, HUD.H),
                                       (0, 0),
                                       (HUD.W, 0),
                                       (HUD.W, HUD.H)])
        sky.set_color(*HUD.SKY_COLOR)
        self.viewer.add_geom(sky)

    def render(self, mode):

        # Get state from environment's dynamics
        dynamics = self.env.dynamics
        state = dynamics.getState()

        # Extract pitch, roll, heading, converting them from radians to degrees
        pitch, roll, heading = np.degrees(state[6:12:2])

        # Get center coordinates
        cx, cy = HUD.W/2, HUD.H/2

        # Center vertical of ground depends on pitch
        gcy = (HUD.H/2 +
               pitch * HUD.PITCH_RETICLE_SPACING / HUD.PITCH_RETICLE_INCREMENT)

        # Left and right top of ground quadrilateral depend on roll
        dx, dy = HUD._rotate(HUD.W, 0, roll)
        x1 = cx - dx
        y1 = gcy - dy
        x2 = cx + dx
        y2 = gcy + dy

        # Draw new ground quadrilateral
        self.viewer.draw_polygon([(x1, y1),
                                  (x2, y2),
                                  (x2, y2-2*HUD.H),
                                  (x1, y1-2*HUD.H)],
                                 color=HUD.GROUND_COLOR)

        # Add a reticle for pitch, rotated by roll to match horizon
        for i in range(-3, 4):

            x1 = 0
            y1 = i * HUD.PITCH_RETICLE_SPACING

            # Alternate the line lengths
            x2 = (x1 +
                  HUD.PITCH_RETICLE_WIDTH + (1-(i % 2)) *
                  HUD.PITCH_RETICLE_WIDTH/2)
            y2 = y1

            x1r, y1r = HUD._rotate(x1, y1, roll)
            x2r, y2r = HUD._rotate(x2, y2, roll)

            # Draw two sets of lines for thickness
            self.viewer.draw_line((cx+x1r, cy+y1r),
                                  (cx+x2r, cy+y2r),
                                  color=HUD.LINE_COLOR)
            self.viewer.draw_line((cx+x1r, cy+y1r+1),
                                  (cx+x2r, cy+y2r+1),
                                  color=HUD.LINE_COLOR)
            self.viewer.draw_line((cx-x1r, cy-y1r),
                                  (cx-x2r, cy-y2r),
                                  color=HUD.LINE_COLOR)
            self.viewer.draw_line((cx-x1r, cy-y1r+1),
                                  (cx-x2r, cy-y2r+1),
                                  color=HUD.LINE_COLOR)

            # Add a label on the left of every other tick
            if i % 2 == 0:
                txt = ('%+3d' % (-i*HUD.PITCH_RETICLE_INCREMENT)).center(3)
                pitch_label = Label(txt,
                                    font_size=HUD.FONT_SIZE,
                                    color=(*HUD.FONT_COLOR, 255),
                                    anchor_x='center',
                                    anchor_y='center')
                label_x = cx-x2r-HUD.PITCH_LABEL_X_OFFSET
                label_y = cy-y2r-HUD.PITCH_LABEL_Y_OFFSET
                HUD._add_label_rotated(self.viewer,
                                       pitch_label,
                                       label_x,
                                       label_y,
                                       roll)

        # Add a horizontal line and center box at the top for the heading
        # display
        y = HUD.H-HUD.HEADING_LINE_Y_OFFSET
        self.viewer.draw_line((0, y), (HUD.W, y), color=HUD.LINE_COLOR)
        self.viewer.draw_polygon([
            (HUD.W/2-HUD.HEADING_BOX_WIDTH, y),
            (HUD.W/2+HUD.HEADING_BOX_WIDTH, y),
            (HUD.W/2+HUD.HEADING_BOX_WIDTH, HUD.H),
            (HUD.W/2-HUD.HEADING_BOX_WIDTH, HUD.H)],
            color=HUD.HIGHLIGHT_COLOR)

        # Display heading
        d = HUD.HEADING_TICK_SPACING * HUD.HEADING_TICK_COUNT
        for i in range(HUD.HEADING_TICK_COUNT):

            HUD._add_label(self.viewer,
                           Label((('%d' %
                                   (i*360//HUD.HEADING_TICK_COUNT)).center(3)),
                                 font_size=HUD.FONT_SIZE,
                                 x=((HUD.W/2 - heading*d/360 +
                                     HUD.HEADING_TICK_SPACING*i) % d),
                                 y=HUD.H-HUD.HEADING_LABEL_Y_OFFSET,
                                 color=(*HUD.FONT_COLOR, 255),
                                 anchor_x='center',
                                 anchor_y='center'))

        # Display altitude at right (negate to accommodate NED)
        HUD._vertical_display(self.viewer,
                              HUD.W-HUD.VERTICAL_BOX_WIDTH,
                              HUD.W-HUD.VERTICAL_BOX_WIDTH+1,
                              -state[4],
                              'Alt (m)')

        # Display ground speed at left
        groundspeed = np.sqrt(state[1]**2 + state[3]**2)
        HUD._vertical_display(self.viewer,
                              10,
                              -HUD.VERTICAL_POINTER_HEIGHT,
                              groundspeed,
                              'GS (m/s)')

        # Add a reticle at the top for roll
        angles = np.linspace(np.radians(180-HUD.ROLL_RETICLE_LIM),
                             np.radians(HUD.ROLL_RETICLE_LIM),
                             HUD.ROLL_RETICLE_PTS)
        points = [(np.cos(a)*HUD.ROLL_RETICLE_RADIUS+HUD.W/2,
                   np.sin(a)*HUD.ROLL_RETICLE_RADIUS+HUD.ROLL_RETICLE_YOFF)
                  for a in angles]
        self.viewer.draw_polyline(points, color=HUD.LINE_COLOR, linewidth=2)
        tickvals = np.append(-np.array(HUD.ROLL_RETICLE_TICKVALS[::-1]),
                             [0] + HUD.ROLL_RETICLE_TICKVALS)
        for tickval in tickvals:
            k = HUD._tickval2index(tickval, tickvals)
            x1, y1 = points[k]
            x2, y2 = x1, y1+HUD.ROLL_RETICLE_TICKLEN
            rangle = (-HUD.ROLL_RETICLE_TICKVALS[-1] /
                      HUD.ROLL_RETICLE_LIM*tickval)
            xr, yr = HUD._rotate(0, HUD.ROLL_RETICLE_TICKLEN, rangle)
            self.viewer.draw_line((x1, y1),
                                  (x2+xr, y2+yr),
                                  color=HUD.LINE_COLOR)
            # Add another tick line for thickness
            self.viewer.draw_line((x1+1, y1),
                                  (x2+xr+1, y2+yr),
                                  color=HUD.LINE_COLOR)

            roll_label = Label(('%2d' % abs(tickval)).center(3),
                               font_size=HUD.FONT_SIZE,
                               color=(*HUD.FONT_COLOR, 255),
                               anchor_x='center',
                               anchor_y='center')

            label_x = x2
            label_y = y2 + HUD.ROLL_RETICLE_TICK_YOFF
            HUD._add_label_rotated(self.viewer,
                                   roll_label,
                                   label_x,
                                   label_y,
                                   rangle/2,
                                   -(6 if rangle == 0 else rangle/3.82))

        # Add a rotated pointer below the current angle in the roll reticle
        x, y = points[HUD._tickval2index(roll, tickvals)]
        x1, y1 = HUD._rotate(-HUD.ROLL_POINTER_SIZE, 0, -roll)
        x2, y2 = HUD._rotate(HUD.ROLL_POINTER_SIZE, 0, -roll)
        x3, y3 = HUD._rotate(0, HUD.ROLL_POINTER_SIZE, -roll)
        y -= HUD.ROLL_POINTER_SIZE
        self.viewer.draw_polygon([(x+x1, y+y1),
                                 (x+x2, y+y2),
                                 (x+x3, y+y3)],
                                 color=HUD.POINTER_COLOR)

        # Add a time display at bottom
        HUD._add_label(self.viewer,
                       Label('Time: %3.2f' % dynamics.getTime(),
                             x=HUD.TIME_LABEL_X, y=HUD.TIME_LABEL_Y,
                             font_size=HUD.LARGE_FONT_SIZE,
                             color=(*HUD.FONT_COLOR, 255),
                             anchor_x='center',
                             anchor_y='center'))

        return self.viewer.render(return_rgb_array=True)

    def close(self):

        return

    def isOpen(self):

        return self.viewer.isopen
