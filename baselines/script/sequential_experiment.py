#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 23:41:36 2018

@author: matteo
"""

import subprocess
import argparse



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seeds', help='RNG seed', type=str, default='109 904 160 570')
parser.add_argument('--script', help='experiment to run', type=str, default='run_poispe_rllab_mujoco')
parser.add_argument('--path', help='where to save', type=str, default='../script/temp')
parser.add_argument('--env', help='task name', type=str, default='cartpole')
parser.add_argument('--delta', help='delta', type=str, default='0.2')
args = parser.parse_args()

def execute(cmd):
    popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ''):
        yield stdout_line 
    popen.stdout.close()
    popen.wait()

for seed in map(int, args.seeds.split(' ')):
    print('SEED %d' % seed)
    for script in execute(['python3 ' + args.script + '.py --seed %d --path %s --delta %s --env %s' % (seed, args.path, args.delta, args.env)]):
        print(script, end='')
