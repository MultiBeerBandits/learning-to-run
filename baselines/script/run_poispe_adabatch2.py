#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:36:59 2018

@author: matteo
"""

import gym
import baselines.envs.continuous_cartpole
import baselines.envs.lqg1d
from baselines.policy.pemlp_policy import PeMlpPolicy
import baselines.pgpe.poisnpe_adabatch2 as poisnpe
import numpy as np

import baselines.common.tf_util as U
sess = U.single_threaded_session()
sess.__enter__()

envs = {'cartpole': 'ContCartPole-v0',
        'lqg': 'LQG1D-v0',
        }

horizons = {'cartpole': 200,
            'lqg': 200,
            }

iters = {'cartpole': 100,
         'lqg': 100,
         }

#Seeds: 107, 583, 850, 730, 808
print('ADABATCH 1')
def train(seed, env_name, shift, normalize, use_rmax, use_renyi, use_parabola, path):
    index = int(str(int(use_parabola)) + str(int(shift)) + str(int(normalize)) + str(int(use_rmax)) + str(int(use_renyi)), 2)
    DIR = path + '/poisnpe/bound_' + str(index) + '/' + env_name + '/seed_' + str(seed)
    import os
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    
    env = gym.make(envs[env_name])
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
              task_horizon=horizons[env_name],
              max_iterations=iters[env_name],
              save_to=DIR,
              verbose=2,
              feature_fun=np.ravel,
              normalize=normalize,
              use_rmax=use_rmax,
              use_renyi=use_renyi,
              max_offline_ite=10,
              max_search_ite=30,
              delta=0.2,
              shift=shift,
              use_parabola=use_parabola)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='save here', type=str, default='temp')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--use_parabola', help='Use parabolic line search (or binary)?', type=int, default=0)
    parser.add_argument('--shift', help='Normalize return?', type=int, default=0)
    parser.add_argument('--normalize', help='Normalize weights?', type=int, default=1)
    parser.add_argument('--use_rmax', help='Use Rmax in bound (or var)?', type=int, default=1)
    parser.add_argument('--use_renyi', help='Use Renyi in ESS (or weight norm)?', type=int, default=1)
    parser.add_argument('--env', help='Environment (RL task)', type=str, default='lqg')
    args = parser.parse_args()
    train(args.seed, args.env, 
          args.shift,
          args.normalize, 
          args.use_rmax,
          args.use_renyi,
          args.use_parabola,
          args.path)
