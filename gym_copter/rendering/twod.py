'''
2D quadcopter rendering

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from gym.envs.classic_control import rendering
import Box2D
from Box2D.b2 import fixtureDef, polygonShape
from gym_copter.dynamics import MultirotorDynamics as dynamics

class TwoDRenderer:

    VIEWPORT_W, VIEWPORT_H = 600, 400
    SCALE                  = 30.0
    GROUND_Z               = 3.33
    GEAR_HEIGHT            = 0.85

    LANDER_POLY = [ (-14, +17), (-17, 0), (-17 ,-10), (+17, -10), (+17, 0), (+14, +17) ]
    HULL_POLY   = [ (-30, 0), (-4, +4), (+4, +4), (+30,  0), (+4, -14), (-4, -14), ]

    LEG_X, LEG_Y, LEG_W, LEG_H         = 12, -7, 3, 20
    MOTOR_X, MOTOR_Y, MOTOR_W, MOTOR_H = 25, 7, 4, 5
    BLADE_X, BLADE_Y, BLADE_W, BLADE_H = 25, 8, 20, 2

    SKY_COLOR     = 0.5, 0.8, 1.0
    GROUND_COLOR  = 0.5, 0.7, 0.3
    VEHICLE_COLOR = 1.0, 1.0, 1.0
    MOTOR_COLOR   = 0.5, 0.5, 0.5
    PROP_COLOR    = 0.0, 0.0, 0.0
    OUTLINE_COLOR = 0.0, 0.0, 0.0

    def __init__(self):

        self.viewer = rendering.Viewer(self.VIEWPORT_W, self.VIEWPORT_H)
        self.viewer.set_bounds(0, self.VIEWPORT_W/self.SCALE, 0, self.VIEWPORT_H/self.SCALE)
        self.world = Box2D.b2World()

        self.lander = self.world.CreateDynamicBody (

                fixtures = [

                    fixtureDef(shape=polygonShape(vertices=[(x/self.SCALE, y/self.SCALE) for x, y in poly]), density=0.0)

                    for poly in [self.HULL_POLY, self._leg_poly(-1), self._leg_poly(+1), 
                        self._motor_poly(+1), self._motor_poly(-1),
                        self._blade_poly(+1,-1), self._blade_poly(+1,+1), self._blade_poly(-1,-1), self._blade_poly(-1,+1)]
                    ]
                )

        # By showing props periodically, we can emulate prop rotation
        self.props_visible = 0

    def close(self):
        self.viewer.close()
        self.world.DestroyBody(self.lander)
        self.lander = None

    def render(self, pose, spinning):

        # Draw ground as background
        self.viewer.draw_polygon(
            [(0,0), 
            (self.VIEWPORT_W,0), 
            (self.VIEWPORT_W,self.VIEWPORT_H), 
            (0,self.VIEWPORT_H)], 
            color=self.GROUND_COLOR)

        # Draw sky
        self.viewer.draw_polygon(
            [(0,self.GROUND_Z), 
            (self.VIEWPORT_W,self.GROUND_Z), 
            (self.VIEWPORT_W,self.VIEWPORT_H), 
            (0,self.VIEWPORT_H)], 
            color=self.SKY_COLOR)

        # Set copter pose to values from Lander2D.step(), negating for coordinate conversion
        self.lander.position = pose[0] + self.VIEWPORT_W/self.SCALE/2, -pose[1] + self.GROUND_Z + self.GEAR_HEIGHT
        self.lander.angle = -pose[2]

        # Draw copter
        self._show_fixture(1, self.VEHICLE_COLOR)
        self._show_fixture(2, self.VEHICLE_COLOR)
        self._show_fixture(0, self.VEHICLE_COLOR)
        self._show_fixture(3, self.MOTOR_COLOR)
        self._show_fixture(4, self.MOTOR_COLOR)

        # Simulate spinning props by alternating show/hide
        if not spinning or self.props_visible: 
            for k in range(5,9):
                self._show_fixture(k, self.PROP_COLOR)

        self.props_visible =  (not spinning or ((self.props_visible + 1) % 3))

    def complete(self, mode):

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _show_fixture(self, index, color):
        fixture = self.lander.fixtures[index]
        trans = fixture.body.transform
        path = [trans*v for v in fixture.shape.vertices]
        self.viewer.draw_polygon(path, color=color)
        path.append(path[0])
        self.viewer.draw_polyline(path, color=self.OUTLINE_COLOR, linewidth=1)

    def _blade_poly(self, x, w):
        return [
            (x*self.BLADE_X,self.BLADE_Y),
            (x*self.BLADE_X+w*self.BLADE_W/2,self.BLADE_Y+self.BLADE_H),
            (x*self.BLADE_X+w*self.BLADE_W,self.BLADE_Y),
            (x*self.BLADE_X+w*self.BLADE_W/2,self.BLADE_Y-self.BLADE_H),
        ]

    def _motor_poly(self, x):
        return [
            (x*self.MOTOR_X,self.MOTOR_Y),
            (x*self.MOTOR_X+self.MOTOR_W,self.MOTOR_Y),
            (x*self.MOTOR_X+self.MOTOR_W,self.MOTOR_Y-self.MOTOR_H),
            (x*self.MOTOR_X,self.MOTOR_Y-self.MOTOR_H)
        ]

    def _leg_poly(self, x):
        return [
            (x*self.LEG_X,self.LEG_Y),
            (x*self.LEG_X+self.LEG_W,self.LEG_Y),
            (x*self.LEG_X+self.LEG_W,self.LEG_Y-self.LEG_H),
            (x*self.LEG_X,self.LEG_Y-self.LEG_H)
        ]
        
        
class TwoDLanderRenderer(TwoDRenderer):

    FLAG_COLOR = 0.8, 0.0, 0.0

    def __init__(self, landing_radius):

        TwoDRenderer.__init__(self)

        self.landing_radius = landing_radius

    def render(self, mode, pose, spinning):

        TwoDRenderer.render(self, pose, spinning)

        # Draw flags
        for d in [-1,+1]:
            flagy1 = self.GROUND_Z
            flagy2 = flagy1 + 50/self.SCALE
            x = d*self.landing_radius + self.VIEWPORT_W/self.SCALE/2
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            self.viewer.draw_polygon([(x, flagy2), (x, flagy2-10/self.SCALE), (x + 25/self.SCALE, flagy2 - 5/self.SCALE)],
                                     color=self.FLAG_COLOR)

        return TwoDRenderer.complete(self, mode)


