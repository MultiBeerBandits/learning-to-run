#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 22:20:25 2018

@author: matteo
"""
from screener import Screener
import random

deltas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
deltas = map(str, deltas)
path = 'tuning'
env = 'swimmer'
script = 'run_poispe_rllab_mujoco_par'

commands = ['python3 sequential_experiment.py --path %s --delta %s --env %s --script %s' % (path + '/delta_' + delta.replace('.',''),
                                                                                    delta, env, script)
                for i, delta in enumerate(deltas)]

Screener().run(commands, name='tuning_swimmer%d' % int(random.random()*1e6))
for c in commands:
    print(c)
