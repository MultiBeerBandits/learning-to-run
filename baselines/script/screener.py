#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 22:20:25 2018

@author: matteo
"""
from multiprocessing import Pool
from screenutils import Screen

class Screener(object):
    
    def command_sender(self, zipped_pair):
        screen, command = zipped_pair
        screen.send_commands(command)

    def run(self, commands, name='s'):
        n_screens = len(commands)
        screens = [Screen(name+'_%d' % (i+1), True) for i in range(n_screens)]
            
        p = Pool(n_screens)
        p.map(self.command_sender, zip(screens, commands))
