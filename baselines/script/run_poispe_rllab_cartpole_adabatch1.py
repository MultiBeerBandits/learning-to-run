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
import baselines.pgpe.poisnpe_adabatch1 as poisnpe
import numpy as np

from baselines.envs.rllab_wrappers import Rllab2GymWrapper
from rllab.envs.box2d.cartpole_env import CartpoleEnv

import baselines.common.tf_util as U
sess = U.single_threaded_session()
sess.__enter__()


def train(seed, shift, normalize, use_rmax, use_renyi, path):
    print('ADABATCH 1')
    index = int(str(int(shift)) + str(int(normalize)) + str(int(use_rmax)) + str(int(use_renyi)), 2)
    DIR = path + '/poisnpe/bound_' + str(index) + '/cartpole_rllab/seed_' + str(seed)
    import os
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    
    env = CartpoleEnv()
    env = Rllab2GymWrapper(env)
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
              task_horizon=500,
              max_iterations=200,
              save_to=DIR,
              verbose=1,
              feature_fun=np.ravel,
              normalize=normalize,
              use_rmax=use_rmax,
              use_renyi=use_renyi,
              max_offline_ite=10,
              max_search_ite=30,
              delta=0.2,
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
    args = parser.parse_args()
    train(args.seed,
          args.shift,
          args.normalize, 
          args.use_rmax,
          args.use_renyi,
          args.path)
