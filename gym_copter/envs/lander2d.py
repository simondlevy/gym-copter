#!/usr/bin/env python3
"""
Copter-Lander, based on https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
"""

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from gym_copter.dynamics.djiphantom import DJIPhantomDynamics

class CopterLander2D(gym.Env, EzPickle):

    # Initial velocity perturbation factor
    INITIAL_RANDOM_VELOCITY = 0.75

    FPS = 50
    SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

    # For rendering after successful landing
    RESTING_DURATION = 50

    # Vehicle display properties ---------------------------------------------------------

    LANDER_POLY =[
        (-14, +17), (-17, 0), (-17 ,-10),
        (+17, -10), (+17, 0), (+14, +17)
        ]

    LEG_X  = 12
    LEG_Y  = -7
    LEG_W  = 3
    LEG_H  = 20

    MOTOR_X  = 25
    MOTOR_Y  = 7
    MOTOR_W  = 4
    MOTOR_H  = 5

    BLADE_X = 25
    BLADE_Y = 8
    BLADE_W = 20
    BLADE_H = 2

    BLADE1L_POLY = [
            (BLADE_X,BLADE_Y),
            (BLADE_X-BLADE_W/2,BLADE_Y+BLADE_H),
            (BLADE_X-BLADE_W,BLADE_Y),
            (BLADE_X-BLADE_W/2,BLADE_Y+-BLADE_H),
            ]

    BLADE1R_POLY = [
            (BLADE_X,BLADE_Y),
            (BLADE_X+BLADE_W/2,BLADE_Y+BLADE_H),
            (BLADE_X+BLADE_W,BLADE_Y),
            (BLADE_X+BLADE_W/2,BLADE_Y+-BLADE_H),
            ]

    BLADE2L_POLY = [
            (-BLADE_X,BLADE_Y),
            (-(BLADE_X+BLADE_W/2),BLADE_Y+BLADE_H),
            (-(BLADE_X+BLADE_W),BLADE_Y),
            (-(BLADE_X+BLADE_W/2),BLADE_Y+-BLADE_H),
            ]

    BLADE2R_POLY = [
            (-BLADE_X,BLADE_Y),
            (-BLADE_X+BLADE_W/2,BLADE_Y+BLADE_H),
            (-BLADE_X+BLADE_W,BLADE_Y),
            (-BLADE_X+BLADE_W/2,BLADE_Y+-BLADE_H),
            ]

    HULL_POLY =[
            (-30, 0),
            (-4, +4),
            (+4, +4),
            (+30,  0),
            (+4, -14),
            (-4, -14),
        ]

    LEG1_POLY = [
            (-LEG_X,LEG_Y),
            (-LEG_X+LEG_W,LEG_Y),
            (-LEG_X+LEG_W,LEG_Y-LEG_H),
            (-LEG_X,LEG_Y-LEG_H)
        ]

    LEG2_POLY = [
            (+LEG_X,LEG_Y),
            (+LEG_X+LEG_W,LEG_Y),
            (+LEG_X+LEG_W,LEG_Y-LEG_H),
            (+LEG_X,LEG_Y-LEG_H)
        ]

    MOTOR1_POLY = [
            (+MOTOR_X,MOTOR_Y),
            (+MOTOR_X+MOTOR_W,MOTOR_Y),
            (+MOTOR_X+MOTOR_W,MOTOR_Y-MOTOR_H),
            (+MOTOR_X,MOTOR_Y-MOTOR_H)
        ]

    MOTOR2_POLY = [
            (-MOTOR_X,MOTOR_Y),
            (-MOTOR_X+MOTOR_W,MOTOR_Y),
            (-MOTOR_X+MOTOR_W,MOTOR_Y-MOTOR_H),
            (-MOTOR_X,MOTOR_Y-MOTOR_H)
        ]


    VIEWPORT_W = 600
    VIEWPORT_H = 400

    TERRAIN_CHUNKS = 11

    SKY_COLOR     = 0.5, 0.8, 1.0
    GROUND_COLOR  = 0.5, 0.7, 0.3
    FLAG_COLOR    = 0.8, 0.0, 0.0
    VEHICLE_COLOR = 1.0, 1.0, 1.0
    MOTOR_COLOR   = 0.5, 0.5, 0.5
    PROP_COLOR    = 0.0, 0.0, 0.0
    OUTLINE_COLOR = 0.0, 0.0, 0.0

    # -------------------------------------------------------------------------------------

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):

        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.prev_reward = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

        # Action is two floats [main engine, left-right engines].
        # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
        # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

        # Helipad
        self.W = self.VIEWPORT_W/self.SCALE
        self.chunk_x = [self.W/(self.TERRAIN_CHUNKS-1)*i for i in range(self.TERRAIN_CHUNKS)]
        self.helipad_x1 = self.chunk_x[self.TERRAIN_CHUNKS//2-1]
        self.helipad_x2 = self.chunk_x[self.TERRAIN_CHUNKS//2+1]
        self.H = self.VIEWPORT_H/self.SCALE
        self.helipad_y = self.H/4

        # Starting position
        self.startpos = self.VIEWPORT_W/self.SCALE/2, self.VIEWPORT_H/self.SCALE

        # Support for rendering
        self.ground = None
        self.lander = None
        self.position = None
        self.angle = None

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.ground: return
        self.world.contactListener = None
        self.world.DestroyBody(self.ground)
        self.ground = None
        self.world.DestroyBody(self.lander)
        self.lander = None

    def reset(self):

        self._destroy()

        self.prev_shaping = None

        self.resting_count = 0

        # Create cusom dynamics model
        self.dynamics = DJIPhantomDynamics()

        # Set its landing altitude
        self.dynamics.setLandingAltitude(self.helipad_y)

        # Initialize custom dynamics with slight velocity perturbation
        state = np.zeros(12)
        d = self.dynamics
        state[d.STATE_Y] =  self.startpos[0] # 3D copter Y comes from 2D copter X
        state[d.STATE_Z] = -self.startpos[1] # 3D copter Z comes from 2D copter Y, negated for NED
        state[d.STATE_Y_DOT] = self.INITIAL_RANDOM_VELOCITY * np.random.randn()
        state[d.STATE_Z_DOT] = self.INITIAL_RANDOM_VELOCITY * np.random.randn()
        self.dynamics.setState(state)

        # By showing props periodically, we can emulate prop rotation
        self.props_visible = 0

        return self.step(np.array([0, 0]))[0]

    def step(self, action):

        # Abbreviation
        d = self.dynamics

        # Stop motors after safe landing
        if self.dynamics.landed() or self.resting_count:
            d.setMotors(np.zeros(4))

        # In air, set motors from action
        else:
            throttle = (action[0] + 1) / 2  # map throttle demand from [-1,+1] to [0,1]
            roll = action[1]
            d.setMotors(np.clip([throttle-roll, throttle+roll, throttle+roll, throttle-roll], 0, 1))
            d.update(1./self.FPS)

        # Get new state from dynamics
        x = d.getState()

        # Parse out state into elements
        posx            =  x[d.STATE_Y]
        posy            = -x[d.STATE_Z] 
        velx            =  x[d.STATE_Y_DOT]
        vely            = -x[d.STATE_Z_DOT]
        angle           = x[d.STATE_PHI]
        angularVelocity = x[d.STATE_PHI_DOT]

        # Set lander pose in display if we haven't landed
        if not (self.dynamics.landed() or self.resting_count):
            self.position = posx, posy
            self.angle = -angle

        # Convert state to usable form
        state = (
            (posx - self.VIEWPORT_W/self.SCALE/2) / (self.VIEWPORT_W/self.SCALE/2),
            (posy - (self.helipad_y)) / (self.VIEWPORT_H/self.SCALE/2),
            velx*(self.VIEWPORT_W/self.SCALE/2)/self.FPS,
            vely*(self.VIEWPORT_H/self.SCALE/2)/self.FPS,
            angle,
            20.0*angularVelocity/self.FPS
            )

        # Shape the reward
        reward = 0
        shaping = 0
        shaping -= 100*np.sqrt(state[0]**2 + state[1]**2)  # Lose points for altitude and vertical drop rate'
        shaping -= 100*np.sqrt(state[2]**2 + state[3]**2)  # Lose points for distance from X center and horizontal velocity
                                                                  
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Assume we're not done yet
        done = False

        # Lose bigly if we go outside window
        if abs(state[0]) >= 1.0:
            done = True
            reward = -100

        elif self.resting_count:

            self.resting_count -= 1

            if self.resting_count == 0:
                done = True

        # It's all over once we're on the ground
        elif self.dynamics.landed():

            # Win bigly we land safely between the flags
            if self.helipad_x1 < posx < self.helipad_x2: 

                reward += 100

                self.resting_count = self.RESTING_DURATION

        elif self.dynamics.crashed():

            # Crashed!
            done = True

        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):

        # Create viewer and world objects if not done yet
        if self.viewer is None:
            self._create_viewer()

        # Draw ground as background
        self.viewer.draw_polygon([(0,0), 
            (self.VIEWPORT_W,0), 
            (self.VIEWPORT_W,self.VIEWPORT_H), 
            (0,self.VIEWPORT_H)], 
            color=self.GROUND_COLOR)

        # Draw sky
        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=self.SKY_COLOR)

        # Draw flags
        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50/self.SCALE
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            self.viewer.draw_polygon([(x, flagy2), (x, flagy2-10/self.SCALE), (x + 25/self.SCALE, flagy2 - 5/self.SCALE)],
                                     color=self.FLAG_COLOR)

        # Set copter pose to values from step()
        self.lander.position = self.position
        self.lander.angle = self.angle

        # Draw copter
        self._show_fixture(1, self.VEHICLE_COLOR)
        self._show_fixture(2, self.VEHICLE_COLOR)
        self._show_fixture(0, self.VEHICLE_COLOR)
        self._show_fixture(3, self.MOTOR_COLOR)
        self._show_fixture(4, self.MOTOR_COLOR)

        # Simulate spinning props by alternating show/hide
        if self.props_visible: 
            for k in range(5,9):
                self._show_fixture(k, self.PROP_COLOR)

        self.props_visible =  self.dynamics.landed() or self.resting_count or ((self.props_visible + 1) % 3)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _show_fixture(self, index, color):
        fixture = self.lander.fixtures[index]
        trans = fixture.body.transform
        path = [trans*v for v in fixture.shape.vertices]
        self.viewer.draw_polygon(path, color=color)
        path.append(path[0])
        self.viewer.draw_polyline(path, color=self.OUTLINE_COLOR, linewidth=1)

    def _create_viewer(self):

        from gym.envs.classic_control import rendering
        import Box2D
        from Box2D.b2 import edgeShape, fixtureDef, polygonShape

        self.viewer = rendering.Viewer(self.VIEWPORT_W, self.VIEWPORT_H)
        self.viewer.set_bounds(0, self.VIEWPORT_W/self.SCALE, 0, self.VIEWPORT_H/self.SCALE)
        self.world = Box2D.b2World()

        # terrain
        height = self.np_random.uniform(0, self.H/2, size=(self.TERRAIN_CHUNKS+1,))
        height[self.TERRAIN_CHUNKS//2-2] = self.helipad_y
        height[self.TERRAIN_CHUNKS//2-1] = self.helipad_y
        height[self.TERRAIN_CHUNKS//2+0] = self.helipad_y
        height[self.TERRAIN_CHUNKS//2+1] = self.helipad_y
        height[self.TERRAIN_CHUNKS//2+2] = self.helipad_y
        smooth_y = [0.33*(height[i-1] + height[i+0] + height[i+1]) for i in range(self.TERRAIN_CHUNKS)]

        self.ground = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (self.W, 0)]))
        self.sky_polys = []
        for i in range(self.TERRAIN_CHUNKS-1):
            p1 = (self.chunk_x[i], smooth_y[i])
            p2 = (self.chunk_x[i+1], smooth_y[i+1])
            self.ground.CreateEdgeFixture(
                vertices=[p1,p2],
                density=0,
                friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], self.H), (p1[0], self.H)])

        self.lander = self.world.CreateDynamicBody (

        position=self.startpos, angle=0.0,

        fixtures = [

            fixtureDef(shape=polygonShape(vertices=[(x/self.SCALE, y/self.SCALE) for x, y in poly]), density=0.0)

            for poly in [self.HULL_POLY, self.LEG1_POLY, self.LEG2_POLY, self.MOTOR1_POLY, self.MOTOR2_POLY,
                self.BLADE1L_POLY, self.BLADE1R_POLY, self.BLADE2L_POLY, self.BLADE2R_POLY]
            ]
        )


def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
                  s[0] is the horizontal coordinate
                  s[1] is the vertical coordinate
                  s[2] is the horizontal speed
                  s[3] is the vertical speed
                  s[4] is the angle
                  s[5] is the angular speed
    returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    # Angle target
    A = 0.5
    B = 3

    # Angle PID
    C = 0.025
    D = 0.05

    # Vertical target
    E = 0.8 

    # Vertical PID
    F = 7.5#10
    G = 10

    angle_targ = s[0]*A + s[2]*B         # angle should point towards center
    angle_todo = (s[4]-angle_targ)*C + s[5]*D

    hover_targ = E*np.abs(s[0])           # target y should be proportional to horizontal offset
    hover_todo = (hover_targ - s[1])*F - s[3]*G

    return hover_todo, angle_todo

def demo_heuristic_lander(env, seed=None, render=False):
    env.seed(seed)
    np.random.seed(seed)
    total_reward = 0
    steps = 0
    state = env.reset()
    while True:
        action = heuristic(env,state)
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if render:
            still_open = env.render()
            if not still_open: break

        if not env.resting_count and (steps % 20 == 0 or done):
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in state]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))

        steps += 1
        if done: break

    env.close()
    return total_reward


if __name__ == '__main__':

    demo_heuristic_lander(CopterLander2D(), seed=None, render=True)
