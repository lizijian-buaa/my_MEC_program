# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 12:49:47 2020

@author: lizijian
"""
from user_equipment import UserEquipment
from task import Task
import constants as cn
import numpy as np
import logging

class server(object):
    def __init__(self, need_preprocess):
        self.f = cn.frequency
        self.f_minportion = cn.f_minportion
        self.f_free = self.f  # unoccupied CPU frequency
        self.apply = None
        self.action_scale = np.array(cn.action_scale)
        self.wait_flag = False
        self.need_preprocess = need_preprocess
        
    def delay(self, task):
        self.f_free -= task.MECS_f
        return task.computation_consumption / task.MECS_f
    
    def offloading_apply(self, task):
        # check wait_flag
        if self.apply == None:
            self.apply = task
            
    def action_preprocess(self, action):
        action = np.clip(action, -.9999999, .9999999)
        action = np.multiply((action.reshape((-1,4))+1)/2, cn.action_scale)
        action[:,:-1] = action[:,:-1].astype(int)
        action[:,-1] = np.maximum(self.f_minportion, action[:,-1])
        return action
        
    def step(self, action, now, BSs, reward):
        if self.need_preprocess:
            action = self.action_preprocess(action)
        reward.reset()
        self.wait_flag = False
        logging.debug('action vector: {}'.format(action))
        reply = action[0] if hasattr(action[0], '__iter__') else action
        task = self.apply
        task.set_work_mode('offload' if reply[0] else 'local')
        task.start_work(now, reward, BSs=BSs, BS=reply[1],
                        channel=reply[2], MECS=self, MECS_f=reply[-1])
    
    
    def get_state(self, BSs, need_preprocess=True):
        # convert the apply into acceptable form:
        x = self.task2np(self.apply)
        state = np.concatenate((x.reshape(-1), \
                                    BSs.busy.reshape(-1).astype(int), \
                                    np.array(self.f_free).reshape(-1)))
        if self.need_preprocess:
            state = state / cn.state_scale
        logging.debug('state vector: {}'.format(state))
        return state
        
    def task2np(self, x):
        # the local info to be transmitted to the MECSn
        x = [x.data_size, x.computation_consumption, x.UE.f,
             x.UE.energy_per_cycle, x.UE.P_send, x.UE.is_sending_occupied,
             x.UE.is_local_occupied, x.UE.zone]
        return np.array(x)
