'''
Mixers for heurisic demos

Copyright (C) 2021 Simon D. Levy

MIT License
'''


def quadxapmix(t, r, p, y):
    '''
    3cw   1ccw
       \ /
        ^
       / \
    2ccw  4cw
    '''

    return [t-r-p-y, t+r+p-y, t+r-p+y, t-r+p+y]


def coaxmix(t, r, p, y):

    return [t-r-p-y, t+r+p-y, t+r-p+y, t-r+p+y]  # XXX
