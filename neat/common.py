'''
Common code for NEAT CopterLanderv2

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import neat
import gym
import numpy as np

class CopterConfig(neat.Config):

    def __init__(self, config_file, reps):

        neat.Config.__init__(self, neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

        self.env = gym.make('gym_copter:Lander-v2')

        self.reps = reps

def eval_genome(genome, config, render=False):

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitness = 0

    config.env.seed(0)

    for _ in range(config.reps):

        state = config.env.reset()
        rewards = 0

        while True:
            action = np.clip(net.activate(state), -1, +1)
            state, reward, done, _ = config.env.step(action)
            if render:
                config.env.render()
            rewards += reward
            if done:
                break

        fitness += rewards

    config.env.close()

    return fitness / config.reps
