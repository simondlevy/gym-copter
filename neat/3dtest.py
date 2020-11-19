#!/usr/bin/env python3
'''
Test script for using NEAT with gym-copter 3D environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import neat
from neat_gym import read_file, eval_net
from gym_copter.rendering.threed import ThreeDLanderRenderer
import threading
import numpy as np
import time

def neat_action(s):

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

    posx, velx, posy, vely, posz, velz, phi, velphi, theta, veltheta = s

    phi_targ = posy*A + vely*B              # angle should point towards center
    phi_todo = (phi-phi_targ)*C + phi*D - velphi*E

    theta_targ = posx*A + velx*B         # angle should point towards center
    theta_todo = -(theta+theta_targ)*C - theta*D  + veltheta*E

    hover_todo = posz*F + velz*G

    return hover_todo, phi_todo, theta_todo # phi affects Y; theta affects X

def neat_lander(net, env):

    total_reward = 0
    steps = 0
    state = env.reset()

    while True:

        action = neat_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if steps % 20 == 0 or done:
           print("observations:", " ".join(["{:+0.2f}".format(x) for x in state]))
           print("step {} total_reward {:+0.2f}".format(steps, total_reward))

        steps += 1

        if done: break

        time.sleep(1./env.FRAMES_PER_SECOND)

    env.close()
    return total_reward


if __name__ == '__main__':

    # Load genome and configuration from pickled file
    genome, config = read_file()

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    renderer = ThreeDLanderRenderer(config.env)

    thread = threading.Thread(target=neat_lander, args=(net, config.env))
    thread.daemon = True
    thread.start()

    # Begin 3D rendering on main thread
    renderer.start() 

    # Run the network
    #print('%6.6f' % eval_net(net, config.env, render=True))
