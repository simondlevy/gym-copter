#!/usr/bin/env python3
'''
3D Copter-Lander with full dynamics (12 state values)

Copyright (C) 2019 Simon D. Levy

MIT License
'''

import numpy as np

from gym_copter.envs.lander3d import Lander3D, heuristic_lander

class TiltedLander3D(Lander3D):

    INITIAL_RANDOM_TILT = 15 # degrees

    def __init__(self):

        Lander3D.__init__(self, initial_random_tilt=self.INITIAL_RANDOM_TILT)

def heuristic(s):
    """
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

    x, dx, y, dy, z, dz, phi, dphi, theta, dtheta = s[:10]

    phi_targ = y*A + dy*B              # angle should point towards center
    phi_todo = (phi-phi_targ)*C + phi*D - dphi*E

    theta_targ = x*A + dx*B         # angle should point towards center
    theta_todo = -(theta+theta_targ)*C - theta*D  + dtheta*E

    hover_todo = z*F + dz*G

    t,r,p = (hover_todo+1)/2, phi_todo, theta_todo  # map throttle demand from [-1,+1] to [0,1]
    return [t-r-p, t+r+p, t+r-p, t-r+p] # use mixer to set motors

if __name__ == '__main__':

    from gym_copter.rendering.threed import ThreeDLanderRenderer
    import threading

    env = TiltedLander3D()

    renderer = ThreeDLanderRenderer(env)

    thread = threading.Thread(target=heuristic_lander, args=(env, heuristic, renderer))
    thread.daemon = True
    thread.start()

    # Begin 3D rendering on main thread
    renderer.start()    
