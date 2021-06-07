'''
Heuristic demo function

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from time import sleep
import numpy as np


def demo_heuristic(env, fun, pidcontrollers, seed=None, csvfilename=None):
    '''
    csvfile arg will only be added by 3D scripts.
    '''

    env.seed(seed)
    np.random.seed(seed)

    total_reward = 0
    steps = 0
    state = env.reset()

    dt = 1. / env.FRAMES_PER_SECOND

    actsize = env.action_space.shape[0]

    csvfile = None
    if csvfilename is not None:
        csvfile = open(csvfilename, 'w')
        csvfile.write('t,' + ','.join([('m%d' % k)
                                      for k in range(1, actsize+1)]))
        csvfile.write(',' + ','.join(env.STATE_NAMES) + '\n')

    while True:

        action = fun(state, pidcontrollers)

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if csvfile is not None:

            csvfile.write('%f' % (dt * steps))

            csvfile.write((',%f' * actsize) % tuple(action))

            csvfile.write(((',%f' * len(state)) + '\n') % tuple(state))

        env.render()

        sleep(1./env.FRAMES_PER_SECOND)

        steps += 1

        if (steps % 20 == 0) or done:
            print('steps =  %04d    total_reward = %+0.2f' %
                  (steps, total_reward))

        if done:
            break

    env.close()

    if csvfile is not None:
        csvfile.close()
