from gym_copter.envs.box2d import CopterBox2D
import numpy as np

class CopterWaypoint(CopterBox2D):

    def __init__(self):

        CopterBox2D.__init__(self, 2, 6)

    def reset(self, xoff):

        return CopterBox2D._reset(self, xoff, 0)

    def _action_to_motors(self, action):

        # Rescale [-1,+1] => [0,1]
        action[0] = (action[0] + 1) / 2 

        # A simple mixer
        motors = [action[0]]*4
        motors[0] += action[1]
        motors[3] += action[1]

        # A simple mixer
        return motors

    def _get_state_reward_done(self):

        pos = self.lander.position
        vel = self.lander.linearVelocity

        state = [
                (pos.x - self.VIEWPORT_W/self.SCALE/2) / (self.VIEWPORT_W/self.SCALE/2),
                (pos.y- (self.helipad_y+self.LEG_H/self.SCALE)) / (self.VIEWPORT_H/self.SCALE/2),
                vel.x*(self.VIEWPORT_W/self.SCALE/2)/self.FPS,
                vel.y*(self.VIEWPORT_H/self.SCALE/2)/self.FPS,
                self.lander.angle,
                20*self.lander.angularVelocity/self.FPS
                ]

        reward = 0

        shaping = - 100*np.sqrt(state[0]**2 + state[1]**2) \
                  - 100*np.sqrt(state[2]**2 + state[3]**2) \
                  - 100*abs(state[4])

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        done = False

        # Quit with a big penalty if we go outside left or right edge of window
        if abs(state[0]) >= 1.0:
            done = True
            reward = -100

        # If we've landed, we're done, with extra reward for a soft landing
        if self.landed:
            if self.ground_count == 0:
                reward += 100 * (abs(state[3]) < self.MAX_LANDING_SPEED)
            else:
                if not self.rendering or self.ground_count == self.GROUND_COUNT_MAX:
                    done = True
            self.ground_count += 1

        return state, reward, done
