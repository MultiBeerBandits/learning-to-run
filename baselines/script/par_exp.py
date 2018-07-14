#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 22:20:25 2018

@author: matteo
"""
from screener import Screener

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', help='save here', type=str, default='../script/temp')
parser.add_argument('--seeds', help='RNG seed', type=str, 
                    default='662 963 100 746 236 247 689 153 947 307 42 950 315 545 178')
parser.add_argument('--env', help='task name', type=str, default='cartpole')
parser.add_argument('--delta', help='delta', type=str, default='0.4')
parser.add_argument('--name', help='name', type=str, default='s')
args = parser.parse_args()

seeds = map(int, args.seeds.split(' '))

commands = ['python3 run_poispe_rllab_mujoco.py --env %s --seed %d --path %s --delta %s' % (args.env, seed,
                                                                                                   args.path,
                                                                                                   args.delta)
                for seed in seeds]

Screener().run(commands, name=args.name)
for c in commands:
    print(c)
