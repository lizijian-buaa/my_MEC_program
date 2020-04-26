# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 22:10:03 2020

this module is meant to create the obj in the simulation environment randomly

@author: lizijian
"""
import constants as cn
import numpy as np
from user_equipment import UserEquipment, random_create_UE
from base_station import BSs
from server import server
from task import Task

class MECsystem(object):
    def __init__(self, num_init_UE, UEnet):
        self.UEs = []
        self.BSs = BSs(cn.BS2MECS_rate, cn.channel_gain, cn.width, cn.noise)
        self.done = False
        self.time = 0
        self.reward = Reward(cn.coefficient_energy, cn.coefficient_time, \
                             cn.fail_punish, cn.battery_empty_punish)
        self.MECS = server(cn.frequency, cn.apply_num, cn.f_minportion, UEnet)
        self.slot = cn.slot
        self.action_c = int(cn.action_slot/self.slot)
        self.initialize()
        
    def initialize(self):
        for i in range(50):
            self.UEs.append(random_create_UE())  # initialized as 50 UEs
        while not self.MECS.apply:
            i = 0
            while i < self.action_c:
                i += 1
                self.time += self.slot
                self.slot_step()
        
    def step(self, action):
        self.MECS.step(action, self.time, self.BSs, self.reward)
        print('time is: {} of {}'.format(self.time, cn.time_total))
        self.MECS.apply.clear()
        while not self.MECS.apply:
            i = 0
            while i < self.action_c:
                i += 1
                self.time += self.slot
                self.slot_step()
            if self.time < cn.time_total:
                self.done = False
        return self.MECS.get_state(self.BSs), self.reward.reward, self.done, {}
    
    def slot_step(self):
        for UE in self.UEs:
            UE.every_slot(self.time, self.slot, self.BSs, self.reward,
                          self.MECS)
    
    def reset(self):
        self.initialize()
        return self.MECS.get_state(self.BSs)

    
class Reward(object):
    def __init__(self, coefficient_energy, coefficient_time, fail_punish,
                 battery_empty):
        self.reward = 0
        self.fail_punish = fail_punish
        self.ce = coefficient_energy
        self.ct = coefficient_time
        self.c_battery_empty = battery_empty
    
    def task_failed(self):
        self.reward += self.fail_punish
        # raise Exception("only for debug")
        # print('任务失败后：reward is {}'.format(self.reward))
    
    def task_finish(self, task):
        self.reward += self.ce*task.energy + self.ct*task.time
        # print('任务完成后：reward is {}'.format(self.reward))
        # print('mode:{}, energy:{}, time:{}'.format(task.work_mode, task.energy,
        #                                               task.time))
        del task
        
    def battery_empty(self):
        self.reward += self.c_battery_empty
        # print('电源没电后：reward is {}'.format(self.reward))
    
    def get_reward(self):
        return self.reward
        
    def reset(self):
        self.reward = 0
        # print('重置后：reward is {}'.format(self.reward))
        
        