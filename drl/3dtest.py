#!/usr/bin/env python3
import threading

import numpy as np
import torch

import gym
from gym import wrappers

from ac_gym import model
from ac_gym.td3 import TD3, eval_policy

from gym_copter.rendering.threed import ThreeDLanderRenderer
from gym_copter.rendering.threed import make_parser, parse


def report(reward, steps):

    print('Got a reward of %+0.3f in %d steps.' % (reward, steps))


def run_td3(parts, env, nhid, record):

    policy = TD3(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            float(env.action_space.high[0]),
            nhid)

    policy.set(parts)

    report(*eval_policy(policy, env, render=(not record), eval_episodes=1))


def run_other(parts, env, nhid, record):

    net = model.ModelActor(env.observation_space.shape[0],
                           env.action_space.shape[0],
                           nhid)

    net.load_state_dict(parts)

    obs = env.reset()

    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor(obs)
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        if np.isscalar(action):
            action = [action]
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    report(total_reward, total_steps)


def main():

    # Make a command-line parser with --view enabled
    parser = make_parser()
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    parser.add_argument('--record', default=None,
                        help='If specified, sets the recording dir')
    parser.add_argument('--seed', default=None, type=int,
                        help='Sets Gym, PyTorch and Numpy seeds')
    args, viewangles = parse(parser)

    # Load network, environment name, and number of hidden units from pickled
    # file
    parts, env_name, nhid = torch.load(open(args.filename, 'rb'))

    # Make a gym environment from the name
    env = gym.make(env_name)

    # Set random seed if indicated
    if args.seed is not None:
        env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Support recordinga  movie
    if args.record:
        env = wrappers.Monitor(env, args.record, force=True)

    # We use a different evaluator functions for TD3 vs. other algorithms
    fun = run_td3 if 'td3' in args.filename else run_other

    # Start the network-evaluation episode on a separate thread
    thread = threading.Thread(target=fun, args=(parts, env, nhid, args.record))
    thread.start()

    # Begin 3D rendering on main thread
    ThreeDLanderRenderer(env, viewangles=viewangles).start()


if __name__ == '__main__':
    main()
