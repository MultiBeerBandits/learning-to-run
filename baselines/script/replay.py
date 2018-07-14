#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:46:33 2018

@author: matteo
"""

import argparse
import gym
import numpy as np
from baselines.policy.pemlp_policy import PeMlpPolicy
import baselines.envs.continuous_cartpole
import baselines.common.tf_util as U

sess = U.single_threaded_session()
sess.__enter__()
pol_file = '../results/npgpe/cartpole/seed_107/weights_99.npy'
horizon = 200
trials = 10
gamma = .99

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', help='RNG seed', type=int, default=None)
parser.add_argument('--pol', help='Algorithm (pgpe, npgpe...)', type=str, default=pol_file)
parser.add_argument('--env', help='Environment (RL task)', type=str, default='cartpole')
args = parser.parse_args()
envs = {'cartpole': 'ContCartPole-v0',
        'lqg': 'LQG1D-v0',
        'swimmer': 'Swimmer-v2',
        'cheetah': 'HalfCheetah-v2',
        }

env = gym.make(envs[args.env])

pol = PeMlpPolicy('pol',
                  env.observation_space, 
                  env.action_space, 
                  hid_layers=[],
                  deterministic=True, 
                  diagonal=True,
                  use_bias=False, 
                  standardize_input=True, 
                  use_critic=False, 
                  seed=args.seed)

rho = np.load(args.pol)
pol.set_params(rho)

for i in range(trials):
    pol.resample()
    print('Higher order params:', pol.eval_params())
    s = env.reset()
    ret = disc_ret = 0
    for t in range(horizon):
        env.render()
        a = pol.act(s)
        s, r, done, _ = env.step(a)
        if done:
            break
        ret+=r
        disc_ret+=gamma**t * r
    print('Ret: %f\nDiscRet: %f' % (ret, disc_ret))