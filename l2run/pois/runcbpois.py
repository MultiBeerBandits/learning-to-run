#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import sys

from baselines.common import (set_global_seeds, boolean_flag)
import gym
import logging
from baselines import logger
from baselines.policy.mlp_policy import MlpPolicy
from baselines.pois import pois
import baselines.common.tf_util as U
import time
from baselines.pois.parallel_sampler import ParallelSampler
from l2run.env_wrapper import create_environment

def train(num_episodes, horizon, iw_method, iw_norm, natural, bound, delta, seed, policy, njobs=1,
         full=False,
          action_repeat = 1,
          reward_scale = 1.,
          fail_reward = -0.2,
          exclude_centering_frame = True,
          integrator_accuracy = 1):

    # Maker function for OpenSim L2Run Environment
    def make_env():
        env = create_environment(False, full, action_repeat, reward_scale,
                                 fail_reward, exclude_centering_frame, integrator_accuracy)
        env.reset()
        env.seed(seed)
        return env

    if policy == 'linear':
        hid_size = num_hid_layers = 0
    elif policy == 'nn':
        hid_size = [64, 64]
        num_hid_layers = 2

    def make_policy(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         hid_size=hid_size, num_hid_layers=num_hid_layers, gaussian_fixed_var=True, use_bias=False, use_critic=False,
                         clip_ob=False, use_rms=False)

    sampler = ParallelSampler(make_policy, make_env, num_episodes, horizon, True, n_workers=njobs, seed=seed)

    sess = U.make_session()
    sess.__enter__()

    set_global_seeds(seed)

    gym.logger.setLevel(logging.WARN)

    pois.learn(make_env, make_policy, n_episodes=num_episodes, max_iters=100,
               horizon=horizon, gamma=1., delta=delta, use_natural_gradient=natural,
               iw_method=iw_method, iw_norm=iw_norm, bound=bound, save_weights=True, sampler=sampler,
               center_return=False)

    sampler.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=423)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=200)
    parser.add_argument('--iw_method', type=str, default='is')
    parser.add_argument('--iw_norm', type=str, default='none')
    parser.add_argument('--natural', type=bool, default=True)
    parser.add_argument('--file_name', type=str, default='progress')
    parser.add_argument('--bound', type=str, default='max-d2')
    parser.add_argument('--delta', type=float, default=0.9)
    parser.add_argument('--njobs', type=int, default=-1)
    parser.add_argument('--policy', type=str, default='nn')
    parser.add_argument('--action-repeat', type=int, default=1)
    boolean_flag(parser, 'full', default=False, help="use full observation")
    boolean_flag(parser, 'exclude-centering-frame', default=False,
                 help="exclude pelvis from observation vec")
    parser.add_argument('--reward-scale', type=float, default=1.0)
    parser.add_argument('--fail-reward', type=float, default=-0.2)
    # Reduce accuracy -> Improve efficiency
    # Good value may be 3e-2
    parser.add_argument('--integrator-accuracy', type=float, default=1e-3)
    args = parser.parse_args()
    logger.configure(dir='.', format_strs=['stdout', 'csv'])
    train(num_episodes=args.num_episodes,
          horizon=args.horizon,
          iw_method=args.iw_method,
          iw_norm=args.iw_norm,
          natural=args.natural,
          bound=args.bound,
          delta=args.delta,
          seed=args.seed,
          policy=args.policy,
          njobs=args.njobs,
          full=args.full,
          action_repeat = args.action_repeat,
          reward_scale = args.reward_scale,
          fail_reward = args.fail_reward,
          exclude_centering_frame = args.exclude_centering_frame,
          integrator_accuracy = args.integrator_accuracy)

if __name__ == '__main__':
    main()