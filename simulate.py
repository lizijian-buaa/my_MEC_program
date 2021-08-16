# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:54:23 2020

@author: lizijian
"""

import numpy as np
from constants import *
from environment import MECsystem
from decision import Agent
import matplotlib.pyplot as plt
import logging
import os


logging.basicConfig(filename='training.log', level=logging.CRITICAL,
                    format='%(levelname)s:%(message)s')
env = MECsystem(API_normalization=True)
MECSnet = Agent(alpha=alpha, beta=beta, input_dims=input_dims, tau=tau,
                env=env, gamma=gamma, batch_size=batch_size,
                layer1_size=layer1_size, layer2_size=layer2_size,
                n_actions=n_actions, lr_decay_rate=lr_decay_rate,
                lr_decay_every=lr_decay_round_length,
                explore_rate=explore_rate)

np.random.seed(0)

done = False
offload_percent, critic_loss, actor_loss, score, k = 0, 0, 0, 0, 0
print_period = 500
score_history = []
obs = env.reset()
print('started')
while not done:
    k += 1
    act = MECSnet.choose_action(obs)
    new_state, reward, done, info = env.step(act)
    MECSnet.remember(obs, act, reward, new_state, int(done), info)
    MECSnet.learn(env.time)  # with decay lr according to simulation time
    score += reward
    obs = new_state
    offload_percent += act[0][0]
    critic_loss += MECSnet.critic_loss
    actor_loss += MECSnet.actor_loss
    # logging.info('reward is： {}'.format(reward))
    if k % print_period == 0:
        print(k)
        print(act)
        print('noise {}'.format(MECSnet.noise()))
        print('slot now {}'.format(env.time/slot))
        offload_percent /= print_period
        print("offloading point {}".format(offload_percent))
        print('already done {} %'.format(env.time/time_total*100))
        score /= print_period
        critic_loss /= print_period
        actor_loss /= print_period
        print('ave_reward {}'.format(score))
        print('ave_critic_loss {}'.format(critic_loss))
        print('ave_actor_loss {}'.format(actor_loss))
        score_history.append(score)
        score, actor_loss, offload_percent, critic_loss = 0, 0, 0, 0
# =============================================================================
#         define a visualization func and call it here
# =============================================================================
    

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
                
