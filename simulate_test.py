# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:16:23 2020

@author: lizijian
"""

import numpy as np
from constants import *
from environment import MECsystem
from decision import Agent
from utils import plot_learning
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='testing.log', level=logging.ERROR,
                    format='%(message)s')

UEnet = Agent(alpha=0.000025, beta=0.00025, input_dims = 8, tau=0.001, \
              env=None, batch_size=64, layer1_size=500, layer2_size=300,
              n_actions=1)
env = MECsystem(apply_num, UEnet)
MECSnet = Agent(alpha=alpha, beta=beta, input_dims=input_dims, tau=0.001,
                env=env, batch_size=batch_size, layer1_size=layer1_size,
                layer2_size=layer2_size, n_actions=n_actions)

np.seed(0)

MECSnet.load_modules()

done = False
reward_history = []
task_fail_count = 0
time_history = []
obs = env.reset()
print('started')
k = 0
while not done:
    k += 1
    act = MECSnet.choose_action(obs, with_noise=False)
    new_state, reward, done, info = env.step(act)
    reward_history.append(reward)
    obs = new_state
    if k % 100 == 0:
        print(k)
        print('already done {} %'.format(env.time/time_total*100)) 

reward_mean = np.mean(reward_history)
reward_var = np.var(reward_history)
delay_mean = np.mean(delay_history)
delay_var = np.var(delay_history)
energy_mean = np.mean(energy_history)
energy_var = np.var(energy_history)

logging.error("module:\nreward:{}(+-{})\ndelay:{}(+-{})\nenergy{}(+-{})\n " \
              .format(reward_mean, reward_var, delay_mean, delay_var,
                      energy_mean, energy_var, task_fail_count))

        
