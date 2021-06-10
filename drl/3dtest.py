#!/usr/bin/env python3
import numpy as np
import torch
import gym

from gym_copter.cmdline import make_parser_3d, wrap

from ac_gym import model
from ac_gym.td3 import TD3, eval_policy

from gym_copter.rendering.threed import ThreeDLanderRenderer


def report(reward, steps, movie):

    print('Got a reward of %+0.3f in %d steps.' % (reward, steps))

    if movie is not None:
        print('Saving movie %s ...' % movie)


def run_td3(env, parts, nhid, movie):

    policy = TD3(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            float(env.action_space.high[0]),
            nhid)

    policy.set(parts)

    report(*eval_policy(policy,
                        env,
                        render=(movie is None),
                        eval_episodes=1),
           movie)


def run_other(parts, env, nhid, movie):

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

    report(total_reward, total_steps, movie)


def main():

    # Make a command-line parser
    parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    parser.add_argument('--movie', default=None,
                        help='If specified, sets the output movie file name')
    parser.add_argument('--seed', default=None, type=int,
                        help='Sets Gym, PyTorch and Numpy seeds')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--hud', action='store_true',
                       help='Use heads-up display')
    group.add_argument('--view', required=False, default='30,120',
                       help='Elevation, azimuth for view perspective')
    args = parser.parse_args()
    viewangles = tuple((int(s) for s in args.view.split(',')))

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

    # We use a different evaluator functions for TD3 vs. other algorithms
    fun = run_td3 if 'td3' in args.filename else run_other

    if args.movie is not None:
        print('Running episode ...')

    # Begin 3D rendering on main thread
    renderer = ThreeDLanderRenderer(env,
                                    fun,
                                    (parts, nhid, args.movie),
                                    viewangles=viewangles,
                                    outfile=args.movie)
    renderer.start()


if __name__ == '__main__':
    main()
