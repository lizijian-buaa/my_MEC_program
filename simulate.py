# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:54:23 2020

@author: lizijian
"""


import numpy as np
from constants import *
from environment import MECsystem
from decision import Agent
from utils import plot_learning

UEnet = Agent(alpha=0.000025, beta=0.00025, input_dims = 8, tau=0.001, \
              env=None, batch_size=64, layer1_size=500, layer2_size=300,
              n_actions=1)
env = MECsystem(apply_num, UEnet)
MECSnet = Agent(alpha=0.000025, beta=0.00025, input_dims = \
              8*apply_num+BS2MECS_rate.size*channel_gain.size+1,
              tau=0.001, env=env, batch_size=64, layer1_size=500,
              layer2_size=300, n_actions=apply_num*4)

np.random.seed(0)

score_history = []
for i in range(10):
    done = False
    score = 0
    obs = env.reset()
    print('started')
    while not done:
        act = MECSnet.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        MECSnet.remember(obs, act, reward, new_state, int(done))
        MECSnet.learn()
        score += reward
        obs = new_state
        print('reward is： {}'.format(reward))
        
    score_history.append(score)
    print('episode ', i, 'score %.2f' % score,
          '100 game average %.2f' % np.mean(score_history[-100:]))
    if i % 25 == 0:
        UEnet.save_models()
    filename = 'MEC_offloading.png'
    plot_learning(score_history, filename, window=100)
        
    
    