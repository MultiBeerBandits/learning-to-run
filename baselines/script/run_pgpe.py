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
import baselines.pgpe.npgpe as npgpe
import baselines.pgpe.pgpe as pgpe
import numpy as np

import baselines.common.tf_util as U
sess = U.single_threaded_session()
sess.__enter__()

envs = {'cartpole': 'ContCartPole-v0',
        'lqg': 'LQG1D-v0',
        'swimmer': 'Swimmer-v2',
        'cheetah': 'HalfCheetah-v2',
        }

algos = {'pgpe': pgpe,
         'npgpe': npgpe,
        }

#Seeds: 107, 583, 850, 730, 808

def train(seed, env_name, algo_name):
    #DIR = './temp'
    DIR = '../results/' + algo_name + '/bias/' + env_name + '/seed_' + str(seed)
    import os
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    
    env = gym.make(envs[env_name])
    env.seed(seed)
    
    pol = PeMlpPolicy('pol',
                      env.observation_space,
                      env.action_space,
                      hid_layers=[],
                      diagonal=True,
                      use_bias=True,
                      standardize_input=True,
                      seed=seed)
    
    algos[algo_name].learn(env,
              pol,
              gamma=1.,
              step_size=1., #1e-2
              batch_size=100,
              task_horizon=200,
              max_iterations=100,
              use_baseline=True,
              step_size_strategy='norm',
              save_to=DIR,
              verbose=2,
              feature_fun=np.ravel)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--algo', help='Algorithm (pgpe, npgpe...)', type=str, default='npgpe')
    parser.add_argument('--env', help='Environment (RL task)', type=str, default='cartpole')
    args = parser.parse_args()
    train(args.seed, args.env, args.algo)
