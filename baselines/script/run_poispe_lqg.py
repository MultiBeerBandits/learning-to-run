#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:36:59 2018

@author: matteo
"""
import sys

from baselines.policy.pemlp_policy import PeMlpPolicy
import baselines.pgpe.poisnpe_adabatch2 as poisnpe
import numpy as np

#Rllab envs

#from rllab.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv

from baselines.envs.lqg1d import LQG1D
import baselines.common.tf_util as U
sess = U.single_threaded_session()
sess.__enter__()


def train(seed, shift, normalize, use_rmax, use_renyi, path, delta):
    DIR = path + '/lqg/seed_' + str(seed)
    import os
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    
    env = LQG1D()
    env.seed(seed)
    
    pol_maker = lambda name: PeMlpPolicy(name,
                      env.observation_space,
                      env.action_space,
                      hid_layers=[],
                      diagonal=True,
                      use_bias=False,
                      seed=seed)
    
    poisnpe.learn(env,
              pol_maker,
              gamma=1.,
              initial_batch_size=100,
              task_horizon=200,
              max_iterations=100,
              save_to=DIR,
              verbose=2,
              feature_fun=np.ravel,
              normalize=normalize,
              use_rmax=use_rmax,
              use_renyi=use_renyi,
              max_offline_ite=10,
              max_search_ite=30,
              delta=delta,
              shift=shift)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='save here', type=str, default='temp')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--shift', help='Normalize return?', type=int, default=0)
    parser.add_argument('--normalize', help='Normalize weights?', type=int, default=1)
    parser.add_argument('--use_rmax', help='Use Rmax in bound (or var)?', type=int, default=1)
    parser.add_argument('--use_renyi', help='Use Renyi in ESS (or weight norm)?', type=int, default=1)
    parser.add_argument('--delta', help='delta', type=str, default='0.2')
    args = parser.parse_args()
    if args.seed is None: args.seed = np.random.randint(low=0, high=999)
    delta = float(args.delta)
    print(delta)
    train(args.seed,
          args.shift,
          args.normalize, 
          args.use_rmax,
          args.use_renyi,
          args.path,
          delta)
