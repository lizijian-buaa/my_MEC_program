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
action_slot = 5e-2  # seconds ---- Handle this carefully!!!
time_total = 50000  # seconds   total simulation time 

# BaseStation
BS2MECS_rate = np.array([0, 0.0001, 0.0002, 0.0003, 0.0004])/KB  # sec/KB
# Channels
channel_gain = np.array([-4, -8, -12, -16])  # dB
channel_gain = np.power(10, channel_gain/20)  # multiple
width = 2*MHZ  # MHz
noise = 4e-8  # W

# MECS
frequency = 20*GHZ  # GHz
f_minportion = 0.1*GHZ  # GHz  # not implemented yet
apply_num = 1  # num of offloading apply input of MECS network

# User Equipments
# Five zones: SBS1, SBS2, SBS3, SBS4 and out of all BS serve
PrUE = np.array([[0, 0.0002, 0.0001, 0.0002, 0.0005],
                 [0.0002, 0, 0.0002, 0.0001, 0.0005],
                 [0.0001, 0.0002, 0, 0.0002, 0.0005],
                 [0.0002, 0.0001, 0.0002, 0, 0.0005],
                 [0.0005, 0.0005, 0.0005, 0.0005, 0]]) * slot
# Transfer Probability Matrix of UE at any slot
UE_frequency = np.array([0.4, 2])*GHZ  # GHz Unified distributed
power_density = np.array([10, 40])/GHZ**2  # (W/GHz^2) Unified distributed
full_Battery = np.array([1000, 4000])*3.7*3.6
# Joule -- mAh*Volt*3.6 Unified distributed
P_send = np.array([3, 8]) # Watt Unified distributed
P_standby = np.array([0.1, 0.2])  # Watt Unified distributed
# Pr(charge_begin) = max(x1 * battery/FullBattery +y1, 0)
# Pr(charge_end) = max(x2 * battery/FullBattery +y2, 0)
x1 = np.array([-0.001, -0.0005])*10
y1 = np.array([3e-4,8e-4])*10
x2 = np.array([0.001, 0.002])
y2 = np.array([-3e-4,-8e-4])
charging_speed_range = np.array([10, 20])  # Watt/second


# Tasks
Prtask = 0.01 * slot  # Probability of task coming at any slot
data_size = np.array([0.2, 10])*MB  # kB Unified distributed
computation_consumption = np.array([1, 50])*GHZ
# CPU_cycle Unified distributed

# training
episode = 2000  # maximum episode
# for the loss function:
coefficient_energy = -0.01  # per Joule
coefficient_time = -0.1  # per second
fail_punish = -20  # per task
battery_empty_punish = -100
delay_tolerant = 50  # seconds
action_scale = [2, BS2MECS_rate.size, channel_gain.size, frequency]


'''
arrival rate            Mbps
arrival data size       Mbps
time slot interval      sec (TBD)
Edge computation cap.   3.3*10^2~10^4
'''
