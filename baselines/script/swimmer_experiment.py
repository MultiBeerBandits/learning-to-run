#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 23:41:36 2018

@author: matteo
"""

import subprocess
import argparse

path = '../results/maythefirst'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seeds', help='RNG seed', type=str, default='107 583 850 730 808')
args = parser.parse_args()

def execute(cmd):
    popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ''):
        yield stdout_line 
    popen.stdout.close()
    popen.wait()

for seed in map(int, args.seeds.split(' ')):
    print('SEED %d' % seed)
    for script in execute(['python3 run_poispe_rllab_swimmer.py --seed %d --path %s' % (seed, path)]):
        print(script, end='')