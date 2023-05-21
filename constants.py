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
PB = 1024*TB

# CPU clock frequency scales
KHZ = 1e3
MHZ = KHZ*1e3
GHZ = MHZ*1e3


# Time scales
slot = 1e-2  # seconds
time_total = 3600 * 24  # seconds   total simulation time

# BaseStation
BS2MECS_rate = np.array([0, 0.0002, 0.0004])/KB  # sec/KB
# Channels
channel_gain = np.array([-4, -12])  # dB
channel_gain = np.power(10, channel_gain/20)  # multiple
width = 2*MHZ  # MHz
noise = 4e-8  # W

# MECS
frequency = 20*GHZ  # GHz
f_minportion = 0.1*GHZ  # GHz  # not implemented yet
apply_num = 1  # num of offloading apply input of MECS network, prefixed

# User Equipments
# Five zones: SBS1, SBS2, SBS3, SBS4 and out of all BS serve
number = 30

PrUE = np.array([[None, 0.001, 0.001, 0.002],
                  [0.001, None, 0.001, 0.002],
                  [0.001, 0.001, None, 0.002],
                  [0.002, 0.002, 0.002, None]])
PrUE[PrUE == None] = 0
PrUE *= 1
PrUE[PrUE == 0] = 1 - np.sum(PrUE, axis=1)
# Transfer Probability Matrix of UE per seconds

move_period = 1  # second(s), prefixed
UE_frequency = np.array([0.5, 4])*GHZ  # GHz Unified distributed
power_density = np.array([1, 4])/GHZ**2  # (W/GHz^2) Unified distributed
full_Battery = np.array([200,1000])*3.7*3.6
# Joule -- mAh*Volt*3.6 Unified distributed
P_send = np.array([5, 15]) # Watt Unified distributed

# Tasks
Prtask = 0.01 * slot  # Probability of task coming at any slot
data_size = np.array([0.2, 10])*MB  # kB Unified distributed
computation_consumption = np.array([1, 50])*GHZ
# CPU_cycle Unified distributed

# training
# episode = 2000  # maximum episode
# for the loss function:
coefficient_energy = -0.005  # per Joule
coefficient_time = -0.1  # per second
finish_reward = 10
fail_punish = -50  # per task
battery_empty_punish = -100
delay_tolerant = 50  # seconds
action_scale = [2, BS2MECS_rate.size, channel_gain.size, frequency]
explore_rate = 0.5

# MECS net
alpha = 1e-8
beta = 1e-7
gamma = 0.99
input_dims = 7*apply_num+BS2MECS_rate.size*channel_gain.size+1
tau = 1e-5
batch_size=64
layer1_size=500
layer2_size=200
n_actions=apply_num*4

'''
arrival rate            Mbps
arrival data size       Mbps
time slot interval      sec (TBD)
Edge computation cap.   3.3*10^2~10^4
'''
