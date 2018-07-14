#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:36:59 2018

@author: matteo
"""
import sys
sys.path.append('/home/alberto/rllab')
sys.path.append('/home/matteo/rllab')

from baselines.policy.pemlp_policy import PeMlpPolicy
import baselines.pgpe.poisnpe_par as poisnpe
import numpy as np
from baselines.pgpe.parallel_sampler import ParallelSampler
from baselines.common import set_global_seeds
import baselines.common.tf_util as U

from baselines.envs.rllab_wrappers import Rllab2GymWrapper
from rllab.envs.mujoco.swimmer_env import SwimmerEnv

gamma = 1.
horizon = 500
iters = 500

#Seeds: 107, 583, 850, 730, 808

def train(seed, env_name, shift, normalize, use_rmax, use_renyi, use_parabola, path, njobs):
    index = int(str(int(use_parabola)) + str(int(shift)) + str(int(normalize)) + str(int(use_rmax)) + str(int(use_renyi)), 2)
    DIR = path + '/poisnpe/bound_' + str(index) + '/' + env_name + '/seed_' + str(seed)
    import os
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    
    def env_maker():
        env = SwimmerEnv()
        env = Rllab2GymWrapper(env)
        env.seed(seed)
        return env
    
    pol_maker = lambda name, ob_space, ac_space: PeMlpPolicy(name,
                      ob_space,
                      ac_space,
                      hid_layers=[],
                      diagonal=True,
                      use_bias=False,
                      seed=seed)
    
    batch_size = 100
    sampler = ParallelSampler(env_maker, pol_maker, gamma, horizon, np.ravel, batch_size, njobs, seed)
    
    sess = U.make_session()
    sess.__enter__()

    set_global_seeds(seed)
    
    poisnpe.learn(env_maker,
              pol_maker,
              sampler,
              batch_size=batch_size,
              task_horizon=horizon,
              max_iterations=iters,
              save_to=DIR,
              verbose=2,
              feature_fun=np.ravel,
              normalize=normalize,
              use_rmax=use_rmax,
              use_renyi=use_renyi,
              max_offline_ite=10, #less is better
              max_search_ite=30,
              delta=0.2,
              shift=shift,
              use_parabola=use_parabola)

    sampler.close()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='save here', type=str, default='temp')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--use_parabola', help='Use parabolic line search (or binary)?', type=int, default=0)
    parser.add_argument('--shift', help='Normalize return?', type=int, default=0)
    parser.add_argument('--normalize', help='Normalize weights?', type=int, default=1)
    parser.add_argument('--use_rmax', help='Use Rmax in bound (or var)?', type=int, default=1)
    parser.add_argument('--use_renyi', help='Use Renyi in ESS (or weight norm)?', type=int, default=1)
    parser.add_argument('--env', help='Environment (RL task)', type=str, default='cartpole')
    parser.add_argument('--njobs', type=int, default=-1)

    args = parser.parse_args()
    
    train(args.seed, args.env, 
          args.shift,
          args.normalize, 
          args.use_rmax,
          args.use_renyi,
          args.use_parabola,
          args.path,
          args.njobs)
