from gym_copter.envs.box2d import CopterBox2D
import numpy as np

class CopterLander1D(CopterBox2D):

    MAX_LANDING_SPEED = 0.05
    GROUND_COUNT_MAX = 15

    def __init__(self):

        CopterBox2D.__init__(self, 2, 1)

    def reset(self, yoff=0):

        return CopterBox2D.reset(self)

    def _action_to_motors(self, action):

        # Rescale [-1,+1] => [0,1]
        action[0] = (action[0] + 1) / 2 

        # A simple mixer
        return [action[0]]*4

    def _get_state_reward_done(self):

        state = [
                (self.lander.position[1] - (self.helipad_y+self.LEG_H/self.SCALE)) / (self.VIEWPORT_H/self.SCALE/2),
                 self.lander.linearVelocity[1] *(self.VIEWPORT_H/self.SCALE/2)/self.FPS
                ]

        reward = 0

        shaping = - 100*np.sqrt(state[0]**2) - 100*np.sqrt(state[1]**2) 

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        done = False

        # If we've landed, we're done, with extra reward for a soft landing
        if self.landed:
            if self.ground_count == 0:
                reward += 100 * (abs(state[1]) < self.MAX_LANDING_SPEED)
            else:
                if not self.rendering or self.ground_count == self.GROUND_COUNT_MAX:
                    done = True
            self.ground_count += 1

        return state, reward, done
