#!/usr/bin/env python3
'''
1D Copter-Lander (single degree of freedom for motion and control)

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from gym_copter.dynamics.djiphantom import DJIPhantomDynamics

class Lander1D(gym.Env, EzPickle):
    
    # Parameters to adjust
    INITIAL_ALTITUDE      = 10
    LANDING_RADIUS        = 2   # for display purposes only
    PENALTY_FACTOR        = 12  # designed so that maximal penalty is around 100
    BOUNDS                = 10
    SAFE_LANDING_BONUS    = 100
    FRAMES_PER_SECOND     = 50

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FRAMES_PER_SECOND
    }

    def __init__(self):

        EzPickle.__init__(self)
        self.viewer = None

        self.prev_reward = None

        # Observation is altitude and its rate of change
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)

        # Action is one value [throttle]
        self.action_space = spaces.Box(-1, +1, (1,), dtype=np.float32)

        # Support for rendering
        self.renderer = None
        self.pose = None

        self.reset()

    def reset(self):

        self._destroy()

        self.prev_shaping = None

        # Create cusom dynamics model
        self.dynamics = DJIPhantomDynamics(self.FRAMES_PER_SECOND)

        # Initialize custom dynamics with random perturbation
        state = np.zeros(12)
        d = self.dynamics
        state[d.STATE_Y] =  0
        state[d.STATE_Z] = -self.INITIAL_ALTITUDE
        self.dynamics.setState(state)

        return self.step(np.array([0]))[0]

    def step(self, action):

        # Abbreviation
        d = self.dynamics
        status = d.getStatus()

        # Stop motors after safe landing
        if status == d.STATUS_LANDED:
            d.setMotors(np.zeros(4))

        # In air, set motors from action
        else:
            t = (action[0]+1)/2  # map throttle demand from [-1,+1] to [0,1]
            d.setMotors(np.clip([t, t, t, t], 0, 1))
            d.update()

        # Get new state from dynamics
        _, _, posy, _, posz, velz, phi, _ = d.getState()[:8]

        # Set lander pose for renderer
        self.pose = posy, posz, phi

        # Convert state to usable form
        state = np.array([posz, velz])

        # Reward is a simple penalty for overall distance and velocity
        shaping = -self.PENALTY_FACTOR * np.sqrt(np.sum(state[0:4]**2))
                                                                  
        reward = (shaping - self.prev_shaping) if (self.prev_shaping is not None) else 0

        self.prev_shaping = shaping

        # Assume we're not done yet
        done = False

        # Lose bigly if we go outside window
        if abs(posy) >= self.BOUNDS:
            done = True

        else:

            # It's all over once we're on the ground
            if status == d.STATUS_LANDED:

                done = True

                reward += self.SAFE_LANDING_BONUS

            elif status == d.STATUS_CRASHED:

                # Crashed!
                done = True

        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):

        from gym_copter.rendering.twod import TwoDLanderRenderer

        # Create renderer if not done yet
        if self.renderer is None:
            self.renderer = TwoDLanderRenderer(self.LANDING_RADIUS)

        d = self.dynamics

        return self.renderer.render(mode, self.pose, d.getStatus())

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _destroy(self):
        if self.renderer is not None:
            self.renderer.close()

# End of Lander1D class ----------------------------------------------------------------


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

    posz, velz = s

    hover_todo = posz*F + velz*G

    return [hover_todo]

def demo_heuristic_lander(env, render=False, save=False):

    from time import sleep

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

        if (steps % 20 == 0) or done:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in state]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))

        steps += 1
        if done: break

    sleep(1)
    env.close()
    return total_reward


if __name__ == '__main__':

    demo_heuristic_lander(Lander1D(), render=True, save=False)
