#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import sys
sys.path.remove('/home/alberto/baselines')
sys.path.append('/home/alberto/baselines_ours')

from baselines.common import set_global_seeds
import gym
import logging
from baselines import logger
from baselines.policy.mlp_policy import MlpPolicy
from baselines.pois import pois
import baselines.common.tf_util as U
import time
from baselines.pois.parallel_sampler import ParallelSampler
from baselines.envs.rllab_wrappers import Rllab2GymWrapper
from baselines.envs.lqg1d import LQG1D

def train(num_episodes, horizon, iw_method, iw_norm, natural, bound, delta, seed, policy, njobs=1):

    def make_env():
        env = gym.make('LQG1D-v0')
        return env

    if policy == 'linear':
        hid_size = num_hid_layers = 0
    elif policy == 'nn':
        hid_size = [100, 50, 25]
        num_hid_layers = 3

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
    parser.add_argument('--policy', type=str, default='linear')
    args = parser.parse_args()
    if args.file_name == 'progress':
        file_name = 'LQG_iw_norm=%s_delta=%s_bound=%s_seed=%s_%s_%s' % (args.iw_norm, args.delta, args.bound, args.seed, args.policy, time.time())
    else:
        file_name = args.file_name
    logger.configure(dir='.', format_strs=['stdout', 'csv'], file_name=file_name)
    train(num_episodes=args.num_episodes,
          horizon=args.horizon,
          iw_method=args.iw_method,
          iw_norm=args.iw_norm,
          natural=args.natural,
          bound=args.bound,
          delta=args.delta,
          seed=args.seed,
          policy=args.policy,
          njobs=args.njobs)

if __name__ == '__main__':
    main()
