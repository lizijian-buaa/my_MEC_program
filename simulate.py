# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:54:23 2020

@author: lizijian
"""


import numpy as np
from constants import *
from environment import MECsystem
from decision import Agent
# from utils import plot_learning
import matplotlib.pyplot as plt
import logging
import os


logging.basicConfig(filename='training.log', level=logging.CRITICAL,
                    format='%(levelname)s:%(message)s')
env = MECsystem(apply_num, UEnet)
MECSnet = Agent(alpha=alpha, beta=beta, input_dims=input_dims, tau=0.001,
                env=env, batch_size=batch_size, layer1_size=layer1_size,
                layer2_size=layer2_size, n_actions=n_actions, gamma=gamma)

np.random.seed(0)

done = False
score = 0
score_history = []
obs = env.reset()
print('started')
k = 0
while not done:
    k += 1
    act = MECSnet.choose_action(obs)
    new_state, reward, done, info = env.step(act)
    MECSnet.remember(obs, act, reward, new_state, int(done), info)
    MECSnet.learn()
    score += reward
    obs = new_state
    # logging.info('reward is： {}'.format(reward))
    if k % 500 == 0:
        print(k)
        print(act)
        print('already done {} %'.format(env.time/time_total*100))
        score_history.append(score/100)
        score = 0

if not os.path.exists("tmp"):
    os.mkdir("tmp")
if not os.path.exists("tmp//ddpg"):
    os.mkdir("tmp//ddpg")
MECSnet.save_models()
np.savetxt("score.dat",score_history)
filename = 'training_reward.png'
plt.figure(1)
plt.plot(score_history)
plt.xlabel('epoch/100')
plt.ylabel('reward')
plt.savefig(filename)
                
