#!/usr/bin/env python3
"""
Rocket trajectory optimization is a classic topic in Optimal Control.

According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).

The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector.
Reward for moving from the top of the screen to the landing pad and zero speed is about 100..140 points.
If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or
comes to rest, receiving an additional -100 or +100 points. Each leg with ground contact is +10 points.
Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame.
Solved is 200 points.

Landing outside the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
on its first attempt. Please see the source code for details.

To see a heuristic landing, run:

python gym/envs/box2d/lunar_lander.py

To play yourself, run:

python examples/agents/keyboard_agent.py LunarLander-v2

Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""

import math
import numpy as np

import Box2D
from Box2D.b2 import edgeShape, fixtureDef, polygonShape

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from gym_copter.dynamics.djiphantom import DJIPhantomDynamics

class LoonieLander(gym.Env, EzPickle):

    FPS = 50
    SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

    MAIN_ENGINE_POWER = 13.0
    SIDE_ENGINE_POWER = 0.6

    INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

    LANDER_POLY =[
        (-14, +17), (-17, 0), (-17 ,-10),
        (+17, -10), (+17, 0), (+14, +17)
        ]

    SIDE_ENGINE_HEIGHT = 14.0
    SIDE_ENGINE_AWAY = 12.0

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

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World(gravity=(0,-9.8665))
        self.ground = None
        self.lander = None

        self.prev_reward = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        # Action is two floats [main engine, left-right engines].
        # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
        # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

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
        self.game_over = False
        self.prev_shaping = None

        W = self.VIEWPORT_W/self.SCALE
        H = self.VIEWPORT_H/self.SCALE

        # terrain
        height = self.np_random.uniform(0, H/2, size=(self.TERRAIN_CHUNKS+1,))
        chunk_x = [W/(self.TERRAIN_CHUNKS-1)*i for i in range(self.TERRAIN_CHUNKS)]
        self.helipad_x1 = chunk_x[self.TERRAIN_CHUNKS//2-1]
        self.helipad_x2 = chunk_x[self.TERRAIN_CHUNKS//2+1]
        self.helipad_y = H/4
        height[self.TERRAIN_CHUNKS//2-2] = self.helipad_y
        height[self.TERRAIN_CHUNKS//2-1] = self.helipad_y
        height[self.TERRAIN_CHUNKS//2+0] = self.helipad_y
        height[self.TERRAIN_CHUNKS//2+1] = self.helipad_y
        height[self.TERRAIN_CHUNKS//2+2] = self.helipad_y
        smooth_y = [0.33*(height[i-1] + height[i+0] + height[i+1]) for i in range(self.TERRAIN_CHUNKS)]

        self.ground = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(self.TERRAIN_CHUNKS-1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i+1], smooth_y[i+1])
            self.ground.CreateEdgeFixture(
                vertices=[p1,p2],
                density=0,
                friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        initial_x = self.VIEWPORT_W/self.SCALE/2
        initial_y = self.VIEWPORT_H/self.SCALE

        # Create dynamics model
        self.dynamics = DJIPhantomDynamics()

        # Start at top center
        state = np.zeros(12)
        #state[self.dynamics.STATE_Y] =  self.START_X + xoff    # 3D copter Y comes from 2D copter X
        #state[self.dynamics.STATE_Z] = -(self.START_Y + yoff)  # 3D copter Z comes from 2D copter Y

        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,

            fixtures = [

                fixtureDef(shape=polygonShape(vertices=[(x/self.SCALE, y/self.SCALE) for x, y in poly]), 
                    density=5.0,
                    friction=0.1,
                    categoryBits=0x0010,
                    maskBits=0x001,   # collide only with ground
                    restitution=0.0)  # 0.99 bouncy

                for poly in [self.HULL_POLY, self.LEG1_POLY, self.LEG2_POLY, self.MOTOR1_POLY, self.MOTOR2_POLY,
                    self.BLADE1L_POLY, self.BLADE1R_POLY, self.BLADE2L_POLY, self.BLADE2R_POLY]
                ]

                )

        self.lander.ApplyForceToCenter( (
            self.np_random.uniform(-self.INITIAL_RANDOM, self.INITIAL_RANDOM),
            self.np_random.uniform(-self.INITIAL_RANDOM, self.INITIAL_RANDOM)
            ), True)

        # By showing props periodically, we can emulate prop rotation
        self.show_props = 0

        return self.step(np.array([0, 0]))[0]

    def step(self, action):

        #print('%+3.3f %+3.3f' % (action[0], action[1]))

        action = np.clip(action, -1, +1).astype(np.float32)

        # Engines
        tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])

        m_power = 0.0
        if action[0] > 0.0:
            # Main engine
            m_power = (np.clip(action[0], 0.0,1.0) + 1.0)*0.5   # 0.5..1.0
            assert m_power >= 0.5 and m_power <= 1.0
            ox =  tip[0] * (4/self.SCALE)
            oy = -tip[1] * (4/self.SCALE)
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            self.lander.ApplyLinearImpulse((-ox * self.MAIN_ENGINE_POWER * m_power, -oy * self.MAIN_ENGINE_POWER * m_power),
                                           impulse_pos,
                                           True)
        s_power = 0.0
        if np.abs(action[1]) > 0.5:
            # Orientation engines
            direction = np.sign(action[1])
            s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
            assert s_power >= 0.5 and s_power <= 1.0
            ox =  side[0] * (direction * self.SIDE_ENGINE_AWAY/self.SCALE)
            oy = -side[1] * (direction * self.SIDE_ENGINE_AWAY/self.SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0] * 17/self.SCALE,
                           self.lander.position[1] + oy + tip[1] * self.SIDE_ENGINE_HEIGHT/self.SCALE)
            self.lander.ApplyLinearImpulse((-ox * self.SIDE_ENGINE_POWER * s_power, -oy * self.SIDE_ENGINE_POWER * s_power),
                                           impulse_pos,
                                           True)

        self.world.Step(1.0/self.FPS, 6*30, 2*30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - self.VIEWPORT_W/self.SCALE/2) / (self.VIEWPORT_W/self.SCALE/2),
            (pos.y - (self.helipad_y)) / (self.VIEWPORT_H/self.SCALE/2),
            vel.x*(self.VIEWPORT_W/self.SCALE/2)/self.FPS,
            vel.y*(self.VIEWPORT_H/self.SCALE/2)/self.FPS,
            self.lander.angle,
            20.0*self.lander.angularVelocity/self.FPS
            ]


        reward = 0
        shaping = 0
        shaping -= 100*np.sqrt(state[0]**2 + state[1]**2)  # Lose points for altitude and vertical drop rate'
        shaping -= 100*np.sqrt(state[2]**2 + state[3]**2)  # Lose points for distance from X center and horizontal velocity
                                                                  
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power*0.30  # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power*0.03

        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done = True
            reward = -100
        if not self.lander.awake:
            done = True
            reward = +100
        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):

        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.VIEWPORT_W, self.VIEWPORT_H)
            self.viewer.set_bounds(0, self.VIEWPORT_W/self.SCALE, 0, self.VIEWPORT_H/self.SCALE)

        # Draw ground as background
        self.viewer.draw_polygon([(0,0), 
            (self.VIEWPORT_W,0), 
            (self.VIEWPORT_W,self.VIEWPORT_H), 
            (0,self.VIEWPORT_H)], 
            color=self.GROUND_COLOR)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=self.SKY_COLOR)

        self._show_fixture(1, self.VEHICLE_COLOR)
        self._show_fixture(2, self.VEHICLE_COLOR)
        self._show_fixture(0, self.VEHICLE_COLOR)
        self._show_fixture(3, self.MOTOR_COLOR)
        self._show_fixture(4, self.MOTOR_COLOR)

        # Simulate spinning props by alernating
        if self.lander.awake and self.show_props:
            for k in range(5,9):
                self._show_fixture(k, self.PROP_COLOR)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50/self.SCALE
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            self.viewer.draw_polygon([(x, flagy2), (x, flagy2-10/self.SCALE), (x + 25/self.SCALE, flagy2 - 5/self.SCALE)],
                                     color=self.FLAG_COLOR)

        self.show_props = (self.show_props + 1) % 3

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

    angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center
    if angle_targ > 0.4: angle_targ = 0.4    # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55*np.abs(s[0])           # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5])*1.0
    hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5

    a = np.array([hover_todo*20 - 1, -angle_todo*20])
    a = np.clip(a, -1, +1)
    return a

def demo_heuristic_lander(env, seed=None, render=False):
    env.seed(seed)
    total_reward = 0
    steps = 0
    s = env.reset()
    while True:
        a = heuristic(env, s)
        s, r, done, info = env.step(a)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open == False: break

        if False: #steps % 20 == 0 or done:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done: break
    env.close()
    return total_reward


if __name__ == '__main__':
    demo_heuristic_lander(LoonieLander(), render=True)
