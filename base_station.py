# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 22:11:19 2020

@author: lizijian
"""
import math
from Node import Node
import numpy as np

class BSs(Node):
    '''
    time1: UE to BS
    time2: BS to MECS
    '''
    def __init__(self, BS2MECS_rate, channel_gains, width, noise):
        self.BS2MECS_rate = BS2MECS_rate
        self.channel_gains = channel_gains
        self.num_BS = self.BS2MECS_rate.size
        self.num_channel = self.channel_gains.size
        self.busy = np.zeros((self.num_BS, self.num_channel), dtype=bool)
        self.noise = noise  # per channel
        self.width = width  # per channel

    def time1(self, UE, task, channel):
        x = UE.P_send * self.channel_gains[int(channel)] / self.noise
        rate = self.width * math.log((1 + x), 2)
        return task.data_size / rate
        
    def time2(self, task, BS):
        return task.data_size * self.BS2MECS_rate[int(BS)]
        
    def delay(self, UE, task, BS, channel):             
        assert not self.busy[int(BS), int(channel)], \
        'channel already been occupied'
        self.set_busy(BS, channel)
        return self.time1(UE, task, channel) + self.time2(task, BS)
        
    def energy(self, UE, task, BS, channel):
        return UE.P_send * self.time1(UE, task, channel)
        '''
        According to many former research, this should include
        UE.P_standby*self.time1(UE, task, channel), while I think without that 
        that part is more reasonable
        '''
        
    def set_busy(self, BS, channel):
        self.busy[int(BS), int(channel)] = True
        
    def set_free(self, task):
        self.busy[int(task.BS_index), int(task.gain_index)] = False
        
    
