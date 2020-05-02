#!/usr/bin/env python3
"""
Adapted from https://raw.githubusercontent.com/openai/gym/master/gym/envs/box2d/lunar_lander.py
"""

import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from gym_copter.dynamics.djiphantom import DJIPhantomDynamics

START_X = 8
START_Y = 13

FPS = 50
SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

INITIAL_RANDOM = 0.2   # Increase to make game harder

LANDER_POLY =[
        (-30, 0),
        (-4, +8),
        (+4, +8),
        (+30,  0),
        (+4, -12),
        (-4, -12),
    ]

LEG_AWAY = -3 
LEG_DOWN = 15
LEG_UP   =  9
LEG_W = 2
LEG_H = 8

VIEWPORT_W = 600
VIEWPORT_H = 400

SKY_COLOR     = 0.5, 0.8, 1.0
GROUND_COLOR  = 0.5, 0.7, 0.3
FLAG_COLOR    = 0.8, 0.0, 0.0
VEHICLE_COLOR = 1.0, 1.0, 1.0
OUTLINE_COLOR = 0.0, 0.0, 0.0

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
        self.leg_contacts = [False, False]

    def BeginContact(self, contact):
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True
                self.leg_contacts[i] = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False

    def bothLegsDown(self):
        return all(self.leg_contacts)

class CopterLander(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
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
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        # Turn off gravity so we can run our own dynamics
        self.world.gravity = 0,0

        # terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H/2, size=(CHUNKS+1,))
        chunk_x = [W/(CHUNKS-1)*i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS//2-1]
        self.helipad_x2 = chunk_x[CHUNKS//2+1]
        self.helipad_y = H/4
        height[CHUNKS//2-2] = self.helipad_y
        height[CHUNKS//2-1] = self.helipad_y
        height[CHUNKS//2+0] = self.helipad_y
        height[CHUNKS//2+1] = self.helipad_y
        height[CHUNKS//2+2] = self.helipad_y
        smooth_y = [0.33*(height[i-1] + height[i+0] + height[i+1]) for i in range(CHUNKS)]

        self.ground = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS-1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i+1], smooth_y[i+1])
            self.ground.CreateEdgeFixture(
                vertices=[p1,p2],
                density=0,
                friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        initial_y = VIEWPORT_H/SCALE
        self.lander = self.world.CreateDynamicBody(
            position=(VIEWPORT_W/SCALE/2, initial_y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[(x/SCALE, y/SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,   # collide only with ground
                restitution=0.0)  # 0.99 bouncy
                )

        self.dynamics = DJIPhantomDynamics()

        state = np.zeros(12)

        # Start at top center
        state[self.dynamics.STATE_Y] = START_X
        state[self.dynamics.STATE_Z] = -START_Y

        # Add a little random noise to initial velocities
        #state[self.dynamics.STATE_Y_DOT]     = INITIAL_RANDOM * np.random.randn()
        #state[self.dynamics.STATE_PHI_DOT] = INITIAL_RANDOM * np.random.randn()

        self.dynamics.setState(state)

        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(VIEWPORT_W/SCALE/2 - i*LEG_AWAY/SCALE, initial_y-LEG_UP),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
            )
            leg.ground_contact = False

            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i*.4, LEG_DOWN/SCALE),
                lowerAngle = 0, 
                upperAngle = 0, 
                enableMotor=True,
                enableLimit=True,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
                )
            leg.joint = self.world.CreateJoint(rjd)

            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        return self.step(np.array([0, 0]))[0]

    def step(self, action):
        '''
        action[0] = throttle strength
        action[1] = roll strength
        '''

        ml_power = 0.0
        mr_power = 0.0

        # Rescale [-1,+1] => [0,1]
        action[0] = (action[0] + 1) / 2 

        motors = [action[0]]*4

        motors[0] += action[1]
        motors[3] += action[1]

        self.dynamics.setMotors(motors)

        self.dynamics.update(1.0/FPS)

        self.world.Step(1.0/FPS, 6*30, 2*30)

        state = self.dynamics.getState()

        # Copy dynamics kinematics out to lander, negating Z for NED => ENU
        dyn = self.dynamics
        self.lander.position        = state[dyn.STATE_Y], -state[dyn.STATE_Z]
        self.lander.angle           = -state[dyn.STATE_PHI]
        self.lander.angularVelocity = -state[dyn.STATE_PHI_DOT]
        self.lander.linearVelocity  = (state[dyn.STATE_Y_DOT], -state[dyn.STATE_Z_DOT])

        pos = self.lander.position
        vel = self.lander.linearVelocity

        state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y- (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel.y*(VIEWPORT_H/SCALE/2)/FPS,
            self.lander.angle,
            20*self.lander.angularVelocity/FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0
            ]
        assert len(state) == 8

        reward = 0
        shaping = \
            - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
            - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
            - 100*abs(state[4]) + 10*state[6] + 10*state[7]  # And ten points for legs contact, the idea is if you
                                                             # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= ml_power*0.30  # less fuel spent is better, about -30 for heuristic landing
        reward -= mr_power*0.03

        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done = True
            reward = -100
        if not self.lander.awake or self.world.contactListener.bothLegsDown():
            done = True
            reward = +100
        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):

        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

        self.viewer.draw_polygon([(0,0), (VIEWPORT_W,0), (VIEWPORT_W,VIEWPORT_H), (0,VIEWPORT_H)], color=GROUND_COLOR)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=SKY_COLOR)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=VEHICLE_COLOR).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=OUTLINE_COLOR, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=VEHICLE_COLOR)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=OUTLINE_COLOR, linewidth=2)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50/SCALE
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            self.viewer.draw_polygon([(x, flagy2), (x, flagy2-10/SCALE), (x + 25/SCALE, flagy2 - 5/SCALE)],
                                     color=FLAG_COLOR)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

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
                  s[6] 1 if first leg has contact, else 0
                  s[7] 1 if second leg has contact, else 0
    returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    throttle_targ = 0.55*np.abs(s[0])           # target y should be proportional to horizontal offset

    roll_todo = s[0]/10
    throttle_todo = (throttle_targ - s[1])*0.25 - (s[3])*.5

    if s[6] or s[7]:  # legs have contact
        roll_todo = 0
        throttle_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

    throttle_todo = throttle_todo*10 - 1

    throttle_todo = np.clip(throttle_todo, -1, +1)

    return np.array([throttle_todo, roll_todo])

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
        if steps % 20 == 0 or done:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done: break
    env.close()
    return total_reward


if __name__ == '__main__':
    demo_heuristic_lander(CopterLander(), render=True)
