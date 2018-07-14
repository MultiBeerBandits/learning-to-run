#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import sys
#sys.path.remove('/home/alberto/baselines')
#sys.path.append('/home/alberto/baselines_ours')
sys.path.append('/home/alberto/rllab')
sys.path.append('/home/matteo/rllab')

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
from rllab.envs.box2d.cartpole_env import CartpoleEnv
import tensorflow as tf

def train(num_episodes, horizon, iw_method, iw_norm, natural, bound, delta, seed, policy, max_offline_iters, njobs=1):

    def make_env():
        env_rllab = CartpoleEnv()
        env = Rllab2GymWrapper(env_rllab)
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

    pois.learn(make_env, make_policy, n_episodes=num_episodes, max_iters=500,
               horizon=horizon, gamma=1., delta=delta, use_natural_gradient=natural,
               iw_method=iw_method, iw_norm=iw_norm, bound=bound, save_weights=True, sampler=sampler,
               center_return=False, render_after=None, max_offline_iters=max_offline_iters)

    sampler.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--iw_method', type=str, default='is')
    parser.add_argument('--iw_norm', type=str, default='none')
    parser.add_argument('--natural', type=bool, default=False)
    parser.add_argument('--file_name', type=str, default='progress')
    parser.add_argument('--bound', type=str, default='max-d2')
    parser.add_argument('--delta', type=float, default=0.9)
    parser.add_argument('--njobs', type=int, default=-1)
    parser.add_argument('--policy', type=str, default='linear')
    parser.add_argument('--max_offline_iters', type=int, default=10)
    args = parser.parse_args()
    if args.file_name == 'progress':
        file_name = 'CARTPOLE_iw_norm=%s_delta=%s_bound=%s_seed=%s_max_offline_iters=%s_%s_%s' % (args.iw_norm, args.delta, args.bound, args.seed, args.policy, args.max_offline_iters, time.time())
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
          max_offline_iters=args.max_offline_iters,
          njobs=args.njobs)

if __name__ == '__main__':
    main()
