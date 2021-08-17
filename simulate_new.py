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
import matplotlib.pyplot as plt
import logging
import os
import plotly_express as px
import xlwt
import xlrd
from xlutils.copy import copy

rb = xlrd.open_workbook('result_data.xlsx')
workbook = copy(rb)
worksheet = workbook.get_sheet(0)
logging.basicConfig(filename='testing.log', level=logging.ERROR,
                    format='%(message)s')
for j in range(len(cn.hypepairs)):
    number, X0 = cn.hypepairs[j]    
    observer = Observer()
    env = MECsystem(API_normalization=False, observer=observer)
    decider = Logical() ############################ 
    # decider = Offloading()
    # decider = Local()
    
    np.random.seed(0)
    
    done = False
    score = 0
    reward_history = []
    print_period = 5000
    obs = env.reset()
    print('started')
    k = 0
    while not done:
        # need some way to 
        k += 1
        act = decider.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        # print('reward {}'.format(reward))
        score += reward
        obs = new_state
        # print('slot now {}'.format(env.time/slot))
        # logging.info('reward is： {}'.format(reward))
        reward_history.append(reward)
        if k % print_period == 0:
            print(k)
            print(act)
            print('slot now {}'.format(env.time/cn.slot))
            print('already done {} %'.format(env.time/cn.time_total*100))
    # =============================================================================
    #         define a visualization func and call it here
    # =============================================================================
        score = 0
    
    # record the ave task_prob, (task_group_num, and zone_num)
    reward_mean = np.mean(reward_history)
    reward_var = np.std(reward_history)
    delay_mean = np.mean(observer.delay)
    delay_var = np.std(observer.delay)
    energy_mean = np.mean(observer.energy)
    energy_var = np.std(observer.energy)
    fail_rate = observer.fail_rate()
    local_count = observer.local_count
    offload_count = observer.offload_count
    
    
    # write to excel
    row = j + 2
    labels = [reward_mean, reward_var, delay_mean, delay_var, energy_mean,
              energy_var, fail_rate, offload_count, local_count]
    worksheet.write(row, 0, label = "{}, {}".format(number, X0))
    for i in range(len(labels)):
        worksheet.write(row, 1+i, label = labels[i])
workbook.save('result_data.xlsx')