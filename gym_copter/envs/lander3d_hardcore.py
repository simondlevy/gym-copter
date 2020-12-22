#!/usr/bin/env python3
'''
3D Copter-Lander with full dynamics (12 state values)

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym_copter.envs.lander3d import Lander3D, heuristic, heuristic_lander

class Lander3DHardcore(Lander3D):

    LANDING_RADIUS        = 2
    INSIDE_RADIUS_BONUS   = 100

    def __init__(self):

        Lander3D.__init__(self)

    def _get_bonus(self, x, y):

        return self.INSIDE_RADIUS_BONUS if x**2+y**2 < self.LANDING_RADIUS**2 else 0

def main():

    from gym_copter.rendering.threed import ThreeDLanderRenderer
    import threading

    env = Lander3DHardcore()

    renderer = ThreeDLanderRenderer(env, radius=2)

    thread = threading.Thread(target=heuristic_lander, args=(env, heuristic, renderer))
    thread.daemon = True
    thread.start()

    # Begin 3D rendering on main thread
    renderer.start()    

if __name__ == '__main__':
    main()
