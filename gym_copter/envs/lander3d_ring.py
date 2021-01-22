#!/usr/bin/env python3
'''
3D Copter-Lander with ring (radius) target

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from gym_copter.envs.lander3d import demo
from gym_copter.envs.lander3d_point import Lander3DPoint


class Lander3DRing(Lander3DPoint):

    LANDING_RADIUS = 2
    INSIDE_RADIUS_BONUS = 100

    def __init__(self):

        Lander3DPoint.__init__(self)

    def _get_bonus(self, x, y):

        return (self.INSIDE_RADIUS_BONUS
                if x**2+y**2 < self.LANDING_RADIUS**2
                else 0)

    def get_radius(self):

        return self.LANDING_RADIUS


class Lander3DRingFixed(Lander3DRing):

    def __init__(self):

        Lander3DRing.__init__(self)

# End of Lander3DRing class ---------------------------------------------------


def heuristic(s):
    '''
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        s (list): The state. Attributes:
                  s[0] is the X coordinate
                  s[1] is the X speed
                  s[2] is the Y coordinate
                  s[3] is the Y speed
                  s[4] is the vertical coordinate
                  s[5] is the vertical speed
                  s[6] is the roll angle
                  s[7] is the roll angular speed
                  s[8] is the pitch angle
                  s[9] is the pitch angular speed
     returns:
         a: The heuristic to be fed into the step function defined above to
            determine the next step and reward.  '''

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

    x, dx, y, dy, z, dz, phi, dphi, theta, dtheta = s[:10]

    phi_targ = y*A + dy*B              # angle should point towards center
    phi_todo = (phi-phi_targ)*C + phi*D - dphi*E

    theta_targ = x*A + dx*B         # angle should point towards center
    theta_todo = -(theta+theta_targ)*C - theta*D + dtheta*E

    hover_todo = z*F + dz*G

    # map throttle demand from [-1,+1] to [0,1]
    t, r, p = (hover_todo+1)/2, phi_todo, theta_todo

    return [t-r-p, t+r+p, t+r-p, t-r+p]  # use mixer to set motors


if __name__ == '__main__':

    demo(Lander3DRing())
