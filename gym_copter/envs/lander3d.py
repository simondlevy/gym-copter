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

    # Perturbation factor for initial horizontal position
    INITIAL_RANDOM_OFFSET = 1.5

    FPS = 50

    SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

    LANDING_RADIUS = 2

    # Rendering properties ---------------------------------------------------------

    # For rendering for a short while after successful landing
    RESTING_DURATION = 50

    LANDER_POLY =[ (-14, +17), (-17, 0), (-17 ,-10), (+17, -10), (+17, 0), (+14, +17) ]

    LEG_X, LEG_Y, LEG_W, LEG_H  = 12, -7, 3, 20

    MOTOR_X, MOTOR_Y, MOTOR_W, MOTOR_H  = 25, 7, 4, 5

    BLADE_X, BLADE_Y, BLADE_W, BLADE_H = 25, 8, 20, 2

    HULL_POLY =[ (-30, 0), (-4, +4), (+4, +4), (+30,  0), (+4, -14), (-4, -14), ]

    VIEWPORT_W = 600
    VIEWPORT_H = 400

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

        # Action is two floats [throttle, roll]
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

        # Ground level
        self.ground_z = self.VIEWPORT_H/self.SCALE/4

        # Support for rendering
        self.ground = None
        self.lander = None
        self.pose = None
        self.angle = None

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.ground: return
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
        self.dynamics.setGroundLevel(self.ground_z)

        # Initialize custom dynamics with random perturbation
        state = np.zeros(12)
        d = self.dynamics
        state[d.STATE_Y] =  self.INITIAL_RANDOM_OFFSET * np.random.randn()
        state[d.STATE_Z] = -self.VIEWPORT_H/self.SCALE
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
            t,r = (action[0]+1)/2, action[1]  # map throttle demand from [-1,+1] to [0,1]
            d.setMotors(np.clip([t-r, t+r, t+r, t-r], 0, 1))
            d.update(1./self.FPS)

        # Get new state from dynamics
        posx, velx, posy, vely, posz, velz, phi, velphi, theta, veltheta = d.getState()[:10]

        # Negate for NED => ENU
        posz  = -posz
        velz  = -velz

        # Set lander pose in display if we haven't landed
        if not (self.dynamics.landed() or self.resting_count):
            self.pose = posy, posz, -phi

        # Convert state to usable form
        state = np.array([
            posy / (self.VIEWPORT_W/self.SCALE/2),
            vely*(self.VIEWPORT_W/self.SCALE/2)/self.FPS,
            (posz - self.ground_z) / (self.VIEWPORT_H/self.SCALE/2),
            velz*(self.VIEWPORT_H/self.SCALE/2)/self.FPS,
            phi,
            20.0*velphi/self.FPS
            ])

        # Reward is a simple penalty for overall distance and velocity
        shaping = -100 * np.sqrt(np.sum(state[0:4]**2))
                                                                  
        reward = (shaping - self.prev_shaping) if (self.prev_shaping is not None) else 0

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
            if abs(posy) < self.LANDING_RADIUS: 

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
        self.viewer.draw_polygon(
            [(0,0), 
            (self.VIEWPORT_W,0), 
            (self.VIEWPORT_W,self.VIEWPORT_H), 
            (0,self.VIEWPORT_H)], 
            color=self.GROUND_COLOR)

        # Draw sky
        self.viewer.draw_polygon(
            [(0,self.ground_z), 
            (self.VIEWPORT_W,self.ground_z), 
            (self.VIEWPORT_W,self.VIEWPORT_H), 
            (0,self.VIEWPORT_H)], 
            color=self.SKY_COLOR)

        # Draw flags
        for d in [-1,+1]:
            flagy1 = self.ground_z
            flagy2 = flagy1 + 50/self.SCALE
            x = d*self.LANDING_RADIUS + self.VIEWPORT_W/self.SCALE/2
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            self.viewer.draw_polygon([(x, flagy2), (x, flagy2-10/self.SCALE), (x + 25/self.SCALE, flagy2 - 5/self.SCALE)],
                                     color=self.FLAG_COLOR)

        # Set copter pose to values from step()
        self.lander.position = self.pose[0] + self.VIEWPORT_W/self.SCALE/2, self.pose[1]
        self.lander.angle = self.pose[2]

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
        from Box2D.b2 import fixtureDef, polygonShape

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
 
# End of CopterLander2D class ----------------------------------------------------------------


def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
                  s[0] is the horizontal coordinate
                  s[1] is the horizontal speed
                  s[2] is the vertical coordinate
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

    posy, vely, posz, velz, phi, velphi = s

    angle_targ = posy*A + vely*B         # angle should point towards center
    angle_todo = (phi-angle_targ)*C + phi*D

    hover_targ = E*np.abs(posy)           # target y should be proportional to horizontal offset
    hover_todo = (hover_targ - posz)*F - velz*G

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
