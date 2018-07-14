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
import baselines.pgpe.poisnpe_adabatch2_par as poisnpe
import numpy as np

#Rllab envs
from baselines.envs.rllab_wrappers import Rllab2GymWrapper
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from baselines.pgpe.batch_parallel_sampler import ParallelSampler

envs = {'cartpole': CartpoleEnv,
        'inverted_pendulum': CartpoleSwingupEnv,
        'mountain_car': MountainCarEnv,
        'acrobot': DoublePendulumEnv}
envs['double_inverted_pendulum'] = InvertedDoublePendulumEnv
envs['swimmer'] = SwimmerEnv
envs['hopper'] = HopperEnv
envs['walker2d'] = Walker2DEnv
envs['half_cheetah'] = HalfCheetahEnv
envs['ant'] = AntEnv
envs['simple_humanoid'] = SimpleHumanoidEnv
envs['humanoid'] = HumanoidEnv

import baselines.common.tf_util as U
from baselines.common import set_global_seeds

def train(seed, shift, normalize, use_rmax, use_renyi, path, env_name, delta):
    DIR = '../results/' + path + '/adapoisnpe/' + env_name +'/seed_' + str(seed)
    import os
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    
    def env_maker():
        env = envs[env_name]()
        env = Rllab2GymWrapper(env)
        return env
        
    pol_maker = lambda name, observation_space, action_space: PeMlpPolicy(name,
                      observation_space,
                      action_space,
                      hid_layers=[],
                      diagonal=True,
                      use_bias=False,
                      seed=seed)

    batch_size = 100
    gamma = 1.
    horizon = 500
    njobs = -1
    sampler = ParallelSampler(env_maker, pol_maker, gamma, horizon, np.ravel, batch_size, njobs, seed)
    
    sess = U.make_session()
    sess.__enter__()

    set_global_seeds(seed)

    
    poisnpe.learn(env_maker,
              pol_maker,
              sampler,
              gamma=1.,
              initial_batch_size=batch_size,
              task_horizon=horizon,
              max_iterations=500,
              save_to=DIR,
              verbose=1,
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
    parser.add_argument('--env', help='task name', type=str, default='cartpole')
    parser.add_argument('--delta', help='delta', type=str, default='0.4')
    args = parser.parse_args()
    delta = float(args.delta)

    if args.seed is None: args.seed = np.random.randint(low=0, high=999)
    train(args.seed,
          args.shift,
          args.normalize, 
          args.use_rmax,
          args.use_renyi,
          args.path,
          args.env,
          delta)
