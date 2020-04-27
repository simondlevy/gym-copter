'''
Third-Person (3D) view using matplotlib for vehicle and target

Copyright (C) 2019 Simon D. Levy

MIT License
'''

from gym_copter.envs.rendering.tpv import TPV

class TpvTarget(TPV):

    def __init__(self, env):

        TPV.__init__(self, env)
        
    def _animate(self, _):

       TPV._animate(self, _)


