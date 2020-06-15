#!/usr/bin/env python3
'''
2D Copter-Lander, based on https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from gym_copter.dynamics.djiphantom import DJIPhantomDynamics

from gym_copter.envs.rendering.twod import TwoDRender

class _TwoDRenderLander(TwoDRender):

    FLAG_COLOR    = 0.8, 0.0, 0.0

    def __init__(self, landing_radius):

        TwoDRender.__init__(self)

        self.landing_radius = landing_radius

    def render(self, mode, pose, landed, resting_count):

        TwoDRender.render(self, pose, landed, resting_count)

        # Draw flags
        for d in [-1,+1]:
            flagy1 = self.GROUND_Z
            flagy2 = flagy1 + 50/self.SCALE
            x = d*self.landing_radius + self.VIEWPORT_W/self.SCALE/2
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            self.viewer.draw_polygon([(x, flagy2), (x, flagy2-10/self.SCALE), (x + 25/self.SCALE, flagy2 - 5/self.SCALE)],
                                     color=self.FLAG_COLOR)

        return TwoDRender.complete(self, mode)

class Lander2D(gym.Env, EzPickle):

    
    # Parameters to adjust
    INITIAL_RANDOM_OFFSET = 1.5 # perturbation factor for initial horizontal position
    INITIAL_ALTITUDE      = 10
    LANDING_RADIUS        = 2
    PENALTY_FACTOR        = 12  # designed so that maximal penalty is around 100
    BOUNDS                = 10
    OUT_OF_BOUNDS_PENALTY = 100
    INSIDE_RADIUS_BONUS   = 100
    RESTING_DURATION      = 1.0 # for rendering for a short while after successful landing
    FRAMES_PER_SECOND     = 50

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FRAMES_PER_SECOND
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

        # Support for rendering
        self.renderer = None
        self.pose = None

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if self.renderer is not None:
            self.renderer.close()

    def reset(self):

        self._destroy()

        self.prev_shaping = None

        self.resting_count = 0

        # Create cusom dynamics model
        self.dynamics = DJIPhantomDynamics()

        # Initialize custom dynamics with random perturbation
        state = np.zeros(12)
        d = self.dynamics
        state[d.STATE_X] =  self.INITIAL_RANDOM_OFFSET * np.random.randn()
        state[d.STATE_Z] = -self.INITIAL_ALTITUDE
        self.dynamics.setState(state)

        return self.step(np.array([0, 0]))[0]

    def step(self, action):

        # Abbreviation
        d = self.dynamics

        # Stop motors after safe landing
        if self.dynamics.landed() or self.resting_count:
            d.setMotors(np.zeros(4))

        # In air, set motors from action
        else:
            t,p = (action[0]+1)/2, action[1]  # map throttle demand from [-1,+1] to [0,1]
            d.setMotors(np.clip([t-p, t+p, t-p, t+p], 0, 1))
            d.update(1./self.FRAMES_PER_SECOND)

        # Get new state from dynamics
        posx, velx, _, _, posz, velz, _, _, theta, veltheta = d.getState()[:10]
 
        # Set lander pose in display if we haven't landed
        if not (self.dynamics.landed() or self.resting_count):
            self.pose = -posx, -posz, theta

        # Convert state to usable form
        state = np.array([posx, velx, posz, velz, theta, veltheta])

        # Reward is a simple penalty for overall distance and velocity
        shaping = -self.PENALTY_FACTOR * np.sqrt(np.sum(state[0:6]**2))
                                                                  
        reward = (shaping - self.prev_shaping) if (self.prev_shaping is not None) else 0

        self.prev_shaping = shaping

        # Assume we're not done yet
        done = False

        # Lose bigly if we go outside window
        if abs(posx) >= self.BOUNDS:
            done = True
            reward = -self.OUT_OF_BOUNDS_PENALTY

        elif self.resting_count:

            self.resting_count -= 1

            if self.resting_count == 0:
                done = True

        # It's all over once we're on the ground
        elif self.dynamics.landed():

            # Win bigly we land safely between the flags
            if abs(posx) < self.LANDING_RADIUS: 

                reward += self.INSIDE_RADIUS_BONUS

                self.resting_count = int(self.RESTING_DURATION * self.FRAMES_PER_SECOND)

        elif self.dynamics.crashed():

            # Crashed!
            done = True

        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):

        # Create viewer and world objects if not done yet
        if self.renderer is None:
            self.renderer = _TwoDRenderLander(self.LANDING_RADIUS)

        return self.renderer.render(mode, self.pose, self.dynamics.landed(), self.resting_count)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

# End of Lander2D class ----------------------------------------------------------------


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
    A = 0.05
    B = 0.06

    # Angle PID
    C = 0.025
    D = 0.05
    E = 0.4

    # Vertical PID
    F = 1.15
    G = 1.33

    posx, velx, posz, velz, theta, veltheta = s

    angle_targ = posx*A + velx*B         # angle should point towards center
    angle_todo = -(theta+angle_targ)*C -theta*D  + veltheta*E

    hover_todo = posz*F + velz*G

    return hover_todo, angle_todo

def demo_heuristic_lander(env, seed=None, render=False, save=False):

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
            frame = env.render('rgb_array')
            if frame is None: break
            if save:
                from PIL import Image
                img = Image.fromarray(frame)
                img.save("img_%05d.png" % steps)

        if not env.resting_count and (steps % 20 == 0 or done):
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in state]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))

        steps += 1
        if done: break

    env.close()
    return total_reward


if __name__ == '__main__':

    demo_heuristic_lander(Lander2D(), seed=None, render=True, save=False)
