#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines.common import set_global_seeds
import gym
import logging
from baselines import logger
from baselines.policy.mlp_policy import MlpPolicy
from baselines.pois import pois
from baselines.envs.continuous_cartpole import CartPoleEnv
import baselines.common.tf_util as U
import ast
import time


def train(num_episodes, horizon, iw_method, iw_norm, natural, bound, ess, seed):

    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    env = gym.make('Hopper-v2')

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=64, num_hid_layers=2, gaussian_fixed_var=True, use_bias=True, use_critic=False)
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    pois.learn(env, policy_fn, num_episodes=num_episodes, iters=500,
               horizon=horizon, gamma=1., delta=0.2, use_natural_gradient='exact' if natural else False,
               iw_method=iw_method, iw_norm=iw_norm, bound=bound, ess_correction=ess)

    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--iw_method', type=str, default='is')
    parser.add_argument('--iw_norm', type=str, default='sn')
    parser.add_argument('--natural', type=bool, default=False)
    parser.add_argument('--file_name', type=str, default='progress')
    parser.add_argument('--bound', type=str, default='student')
    parser.add_argument('--ess', type=ast.literal_eval, default=False)
    args = parser.parse_args()
    if args.file_name == 'progress':
        file_name = 'pois_hopper_iw_method=%s_iw_norm=%s_natural=%s_bound=%s_ess=%s_%s' % (args.iw_method, args.iw_norm, args.natural, args.bound, args.ess, time.time())
    else:
        file_name = args.file_name
    logger.configure(dir='.', format_strs=['stdout', 'csv'], file_name=file_name)
    train(num_episodes=args.num_episodes,
          horizon=args.horizon,
          iw_method=args.iw_method,
          iw_norm=args.iw_norm,
          natural=args.natural,
          bound=args.bound,
          ess=args.ess,
          seed=args.seed)

if __name__ == '__main__':
    main()
