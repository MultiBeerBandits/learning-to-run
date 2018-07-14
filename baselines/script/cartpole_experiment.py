#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 23:41:36 2018

@author: matteo
"""

import subprocess
import argparse



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seeds', help='RNG seed', type=str, default='107 583 850 730 808')
parser.add_argument('--script', help='experiment to run', type=str, default='run_poispe_rllab_cartpole.py')
parser.add_argument('--path', help='where to save', type=str, default='../script/temp')
args = parser.parse_args()
path = '../results/' + args.path

def execute(cmd):
    popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ''):
        yield stdout_line 
    popen.stdout.close()
    popen.wait()

for seed in map(int, args.seeds.split(' ')):
    print('SEED %d' % seed)
    for script in execute(['python3 ' + args.script + ' --seed %d --path %s' % (seed, path)]):
        print(script, end='')