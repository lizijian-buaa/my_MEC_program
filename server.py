1# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 12:49:47 2020

@author: lizijian
"""
from user_equipment import UserEquipment
from task import Task
import constants as cn
import numpy as np

class server(object):
    def __init__(self, all_CPU_frequency, num_UE, f_minportion, UEnet):
        self.f = all_CPU_frequency
        self.f_minportion = f_minportion
        self.f_free = self.f  # unoccupied CPU frequency
        self.apply = []
        self.final_apply = []
        self.num_UE = num_UE  # the number of UEs allowed to make 
        # offloading decision via MECS-net
        self.virtual_apply = np.array([0,0,cn.full_Battery[-1],
                                       cn.full_Battery[-1],cn.UE_frequency[-1],
                                       0, cn.P_send[-1], 1])
        self.action_scale = np.array(cn.action_scale)
        self.UEnet = UEnet
        self.wait_flag = False
        # self.notapply = True
    
    def delay(self, task):
        self.f_free -= task.MECS_f
        return task.computation_consumption / task.MECS_f            
    
    def offloading_apply(self, task):
        # check wait_flag
        if not self.wait_flag:
            self.apply.append(task)
            if len(self.apply) == cn.apply_num:
                self.wait_flag = True

    def deny_lowODs(self):
        return
        if len(self.apply) > self.num_UE:
            for task in self.apply:
                task.set_OD(self.UEnet)
            self.apply.sort(key=lambda x:x.od, reverse=True)
            self.final_apply = self.apply[:cn.apply_num]
            for task in self.apply[cn.apply_num:]:
                task.set_work_mode('local')
            
    def step(self, action, now, BSs, reward):
        # implement action here, action: apply_num * 4, index range:(-1, 1)
        reward.reset()
        self.wait_flag = False
        # print('reply vector: {}'.format(action))
        action = self.preprocess(action)
        for i, task in enumerate(self.apply):
            reply = action[i]
            # print('reply vector: {}'.format(reply))
            task.set_work_mode('offload' if reply[0] else 'local')
            task.start_work(now, reward, BSs=BSs, BS=reply[1],
                            channel=reply[2], MECS=self, MECS_f=reply[-1])
        self.deny_lowODs()
        
    def preprocess(self, action):
        action = np.clip(action, -.9999999, .9999999)
        action = np.multiply((action.reshape((-1,4))+1)/2, self.action_scale)
        action[:,:-1] = action[:,:-1].astype(int)
        action[:,-1] = np.maximum(self.f_minportion, action[:,-1])
        return action
        
    def get_state(self, BSs):   
        # convert the apply into acceptable form:
        x = np.zeros((cn.apply_num, 8))
        for i, task in enumerate(self.apply):
            x[i] = self.task2np(task)
        for i in range(len(self.apply), cn.apply_num):
            x[i] = self.virtual_apply
        state = np.concatenate((x.reshape(-1), \
                                    BSs.busy.reshape(-1).astype(int), \
                                    np.array(self.f_free).reshape(-1)))
        return state
        
    def task2np(self, x):
        x = [x.data_size, x.computation_consumption, x.UE.b, x.UE.B, x.UE.f,
             x.UE.energy_per_cycle, x.UE.P_send, x.UE.ischarging]
        return np.array(x)
    
        