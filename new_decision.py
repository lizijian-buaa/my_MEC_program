#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:21:26 2021

@author: zijian
"""
import constants as cn
import numpy as np
from math import log


class Logical():
    def __init__(self):
        self.useMECS = cn.useMECS
        self.channel_num = cn.channel_gain.size
        self.BS_num = cn.BS2MECS_rate.size
        self.BS = None
        self.channel = None
        self.local = np.array([0, 0, 0, 0])
        
    def choose_action(self, observation):
        # observation looks like:
# =============================================================================
#         task[x.data_size, x.computation_consumption]
#         + 
#             [x.UE.f, x.UE.energy_per_cycle, x.UE.P_send,
#              x.UE.is_sending_occupied, x.UE.is_local_occupied, x.UE.zone]
#         +
#         busy[num_BSs*num_channels]
#         +
#         MECS_free
# =============================================================================
        self.task_info = observation[: 2]
        self.UE_info = observation[2: 8]
        busy = observation[8: self.channel_num*self.BS_num+8]
        self.busy = busy.reshape((self.BS_num, self.channel_num))
        self.MECS_free = observation[-1]
        self.zone = self.UE_info[-1]
        if self.UE_info[3] == True:
            # sending is occupied, run locally, whether local occupied or not
            return self.local
        elif self.UE_info[4] == True:
            # should run offloadingly
            if self.choose_tunnel():
                # a tunnel avaliable can be found
                act = self.get_offload_result(withcost=False)
                return act
            else:
                return self.local
        if not self.choose_tunnel():
            return self.local
        # compare local and offload
        lt = self.task_info[1] / self.UE_info[0]
        le = lt * self.UE_info[0] * self.UE_info[1]
        ot, oe, act = self.get_offload_result(withcost=True)
        accordance = cn.coefficient_time * (ot - lt) + cn.coefficient_energy *\
                     (oe - le)
        if accordance > 0:
            return act
        else:
            return self.local
        
        
    def choose_tunnel(self):
        BSchannel = self.busy
        if not all(BSchannel[int(self.zone)]):
            self.BS = self.zone
        elif not all(BSchannel[0]):
            # use macro BS
            self.BS = 0
        else:
            # no communication tunnel
            return False
        self.channel = np.where(BSchannel[int(self.BS)]==False)[0][0]
            # one of this two [0] is meant to unwrap the output array
        return True
        
    def get_offload_result(self, withcost):
        freq = self.MECS_free * self.useMECS
# =============================================================================
#       action_scale = [2, BS2MECS_rate.size, channel_gain.size, frequency]
# =============================================================================
        act = np.array([1, self.BS, self.channel, freq])
        if withcost:
            x = self.UE_info[2] * cn.channel_gain[int(self.channel)] / cn.noise
            rn = cn.width * log((1 + x), 2)
            t1 = self.task_info[0] / rn
            t2 = self.task_info[0] * cn.BS2MECS_rate[int(self.BS)]
            t3 = self.task_info[1] / freq
            t = t1 + t2 + t3
            e = self.UE_info[2] * t1
            return t, e, act
        else:
            return act
        
        
def preprocess(action):
    action = np.clip(action, -.9999999, .9999999)
    action = np.multiply((action.reshape((-1,4))+1)/2, cn.action_scale)
    action[:,:-1] = action[:,:-1].astype(int)
    action[:,-1] = np.maximum(cn.f_minportion, action[:,-1])
    return action

        
class Offloading(object):
    # use SBS prior to MBS defaultly
    def __init__(self):
        self.useMECS = cn.useMECS
        self.channel_num = cn.channel_gain.size
        self.BS_num = cn.BS2MECS_rate.size
        self.BS = None
        self.channel = None
        self.local = np.array([0, 0, 0, 0])
    
    def choose_action(self, observation):
        # observation looks like:
# =============================================================================
#         task[x.data_size, x.computation_consumption]
#         + 
#             [x.UE.f, x.UE.energy_per_cycle, x.UE.P_send,
#              x.UE.is_sending_occupied, x.UE.is_local_occupied, x.UE.zone]
#         +
#         busy[num_BSs*num_channels]
#         +
#         MECS_free
# =============================================================================
        self.task_info = observation[: 2]
        self.UE_info = observation[2: 8]
        busy = observation[8: self.channel_num*self.BS_num+8]
        self.busy = busy.reshape((self.BS_num, self.channel_num))
        self.MECS_free = observation[-1]
        self.zone = self.UE_info[-1]
        # process of observation

        BSchannel = self.busy
        if not all(BSchannel[int(self.zone)]):
            self.BS = self.zone
        elif not all(BSchannel[0]):
            self.BS = 0
        else:
            # no communication tunnel
            return np.array([1, 0, 0, 0]) # this is an invalid action
        self.channel = np.where(BSchannel[int(self.BS)]==False)[0][0]
            # one of this two [0] is meant to unwrap the output array
        freq = self.MECS_free * self.useMECS
# =============================================================================
#       action_scale = [2, BS2MECS_rate.size, channel_gain.size, frequency]
# =============================================================================
        act = np.array([1, self.BS, self.channel, freq])
        return act
            
class Local():
    def __init__(self):
        pass
        
    def choose_action(self, observation):     
        return np.array([0, 0, 0, 0])
