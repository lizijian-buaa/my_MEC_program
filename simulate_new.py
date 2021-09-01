#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 16:58:15 2021

@author: zijian
"""

import numpy as np
import constants as cn
from environment import MECsystem, Observer
from new_decision import Logical, Offloading, Local
# import matplotlib.pyplot as plt
import logging
# import os
# import plotly_express as px
from write_excel import Write_result

        
np.random.seed(0)
logging.basicConfig(filename='testing.log', level=logging.ERROR,
                    format='%(message)s')
observer = Observer()
env = MECsystem(API_normalization=False, observer=observer)
excel_writer = Write_result('result_data.xlsx')
deciders = {"OFFLOADING": Offloading(), "GREEDY": Logical(), "LOCAL": Local()}

number, X0 = cn.number, cn.X0 
for useMECS in cn.useMECSs:
    for key in list(deciders.keys())[:-1]:
        deciders[key].set_useMECS(useMECS)
        decider = deciders[key]
        excel_writer.write_head(method=key)    

        done = False
        print_period = 5000
        obs = env.reset(None, False, observer)
        print('started')
        k = 0
        while not done:
            k += 1
            act = decider.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            # print('reward {}'.format(reward))
            # print('slot now {}'.format(env.time/slot))
            # logging.info('reward is： {}'.format(reward))
            if k % print_period == 0:
                print(k)
                print(obs)
                print(act)
                print('slot now {}'.format(env.time/cn.slot))
                print('already done {} %'.format(env.time/cn.time_total*100))
            obs = new_state
                
        row_data = ["{}, {}".format(number, X0)] + observer.output_to_excel()
        excel_writer.write_data(row_data)


for key in deciders.keys():
    decider = deciders[key]
    decider.set_useMECS(cn.useMECS)
    excel_writer.write_head(method=key)
    for j in range(len(cn.hypepairs)):
        number, X0 = cn.hypepairs[j]
        
        done = False
        print_period = 5000
        obs = env.reset(j, False, observer)
        print('started')
        k = 0
        while not done:
            k += 1
            act = decider.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            # print('reward {}'.format(reward))
            # print('slot now {}'.format(env.time/slot))
            # logging.info('reward is： {}'.format(reward))
            if k % print_period == 0:
                print(k)
                print(obs)
                print(act)
                print('slot now {}'.format(env.time/cn.slot))
                print('already done {} %'.format(env.time/cn.time_total*100))
        # =============================================================================
        #         define a visualization func and call it here
        # =============================================================================
            obs = new_state
            
        # write to excel
        row_data = ["{}, {}".format(number, X0)] + observer.output_to_excel()
        excel_writer.write_data(row_data)
excel_writer.save()
