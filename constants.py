# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 18:23:43 2020

@author: lizijian
os.chdir('C:\\Users\\lizijian\\Desktop\\edge computing\\my_MEC_program')
locally largest delay:
    computation_consumption[1]/UE_frequency[0]
locally largest energy consumption:
    computation_consumption[1]*power_density[1]*UE_frequency[1]
remotely largest delay:
    data_size[-1]/(width*math.log((1 + P_send[0]* \
    channel_gain[0] / noise), 2))+data_size[-1]*BS2MECS_rate[-1] \
    + computation_consumption[1]/frequency
remotely largest energy consumption:
"""


import math
import numpy as np


# Data size scales
BYTE = 8
KB = 1024*BYTE
MB = 1024*KB
GB = 1024*MB
TB = 1024*GB

# CPU clock frequency scales
KHZ = 1e3
MHZ = KHZ*1e3
GHZ = MHZ*1e3


# Time scales
slot = 1e-1  # seconds of time length per slot
time_total = 3600 * 48  # seconds   total simulation time


# BaseStation
BS2MECS_rate = np.array([0, 0.05, 0.1, 0.15])/MB  # sec/MB
# 0, 0.0002, 0.0004
# Channels
channel_gain = np.array([-5, -10, -15])  # dB -5, -10, -15
channel_gain = np.power(10, channel_gain/20)  # multiple
width = 5*MHZ  # MHz
noise = 4e-8  # W


# MECS
frequency = 30*GHZ  # GHz
f_minportion = 0.1*GHZ  # minimum computing resources if it is to be allocated
apply_num = 1  # num of offloading apply input of MECS network, prefixed


# User Equipments
# Five zones: SBS1, SBS2, SBS3, SBS4 and out of all BS serve
##############################
numbers = [20, 1, 5, 50, 100]
number = 20  # default
##############################

PrUE = np.array([[None, 0.001, 0.001, 0.001],
                  [0.001, None, 0.001, 0.001],
                  [0.001, 0.001, None, 0.001],
                  [0.004, 0.004, 0.004, None]])
PrUE[PrUE == None] = 0
PrUE *= 1
PrUE[PrUE == 0] = 1 - np.sum(PrUE, axis=1)
# Transfer Probability Matrix of UE per seconds
move_period = 1  # second(s), prefixed
change_Prtask_period = 60
changeProbProb = 0.2  # every minute
# move_period for User movement between BSs' svrvice areas and Prtask change
UE_frequency = np.array([0.5, 2])*GHZ  # GHz Unified distributed
power_density = np.array([1, 4])/GHZ**2  # (W/GHz^2) Unified distributed
# Joule -- mAh*Volt*3.6 Unified distributed
P_send = np.array([5, 15]) # Watt Unified distributed


# Tasks
# CPU_cycle Unified distributed
data_size = np.array([0.2, 10])*MB  # kB Unified distributed
computation_consumption = np.array([0.5, 5])*GHZ
##############################\
X0s = [0.01, 0.001, 0.005, 0.02, 0.1]
X0 = 0.01  # default (can be understood as every task coming per sec)
##############################
change_Prtask = {"ins": 2, "outs": 1.5, "up": 0.1*slot, "down": 10*slot,
                 "x0": slot}  # up, down, x0 need to be multiplied by X0
# parameters for change rules
task_group_num = 3
# training
# episode = 2000  # maximum episode
# for the loss function:
# =============================================================================
#     This is the most important for the success of the RL decision stretegy,
#     Try different params to check which is the best
# =============================================================================
coefficient_energy = -0.02  # per Joule
coefficient_time = -0.5  # per second
finish_reward = 10
fail_punish = -4  # per task (try -4 maybe)
delay_tolerant = 10  # seconds
state_scale = np.array([data_size[-1], computation_consumption[-1], 
                        UE_frequency[-1], power_density[-1]*UE_frequency[-1],
                        P_send[-1], 1, 1, PrUE.shape[0]])
state_scale = np.concatenate((state_scale,
                              np.array([1]*BS2MECS_rate.size*
                                       channel_gain.size),
                              np.array([frequency])))
action_scale = np.array([2, BS2MECS_rate.size, channel_gain.size, frequency])
explore_rate = 2

# MECS neural net
alpha = 5e-6  # learning rate of the actor network of ddpg
beta = 1e-7  # learning rate of the critic network of ddpg
gamma = 0.99  # reward decay rate per sec
gamma = pow(gamma, slot)  # reward decay rate per slot
input_dims = 8*apply_num+BS2MECS_rate.size*channel_gain.size+1
tau = 0.999  # network params soft update rate
batch_size=64
layer1_size=500
layer2_size=200
n_actions=apply_num * 4
# =============================================================================
# learning rate and exploration rate decay decaying hyperbolicly to "decay" 
# through the training process.
# =============================================================================
lr_total_decay = 0.1
lr_decay_round = 10
lr_decay_round_length = time_total / lr_decay_round
lr_decay_rate = pow(lr_total_decay, 1/lr_decay_round)  # per slot

useMECS = 2/3
useMECSs = [1/4, 1/2, 3/4, 1]
hypepairs = [(numbers[i], X0s[0]) for i in range(len(numbers))] + \
             [(numbers[0], X0s[i]) for i in range(len(X0s))]


'''
arrival rate            Mbps
arrival data size       Mbps
time slot interval      sec (TBD)
Edge computation cap.   3.3*10^2~10^4
'''
