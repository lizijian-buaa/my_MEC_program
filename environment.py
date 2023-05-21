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
import logging
from random import shuffle

class MECsystem(object):
    def __init__(self, num_init_UE, UEnet):
        self.UEs = []
        self.BSs = BSs(cn.BS2MECS_rate, cn.channel_gain, cn.width, cn.noise)
        self.done = False
        self.time = 0
        self.reward = Reward(cn.coefficient_energy, cn.coefficient_time, \
                             cn.fail_punish, cn.battery_empty_punish)
        self.MECS = server(cn.frequency, cn.apply_num, cn.f_minportion, UEnet, 
                           cn.change_slot_num)
        self.slot = cn.slot
        self.move_clock = cn.move_period
        self.initialize()
        
    def initialize(self):
        for i in range(cn.number):
            self.UEs.append(random_create_UE())  # initialized UEs
        while not self.MECS.apply:
            self.time += self.slot
            self.slot_step()
        
    def step(self, action, train = True):
        i = 0
        self.MECS.step(action, self.time, self.BSs, self.reward)
        logging.info('time is: {} of {}'.format(self.time, cn.time_total))
        self.MECS.apply.clear()
        while not self.MECS.apply:
            self.time += self.slot
            i += 1
            self.slot_step()
        if self.time > cn.time_total:
            self.done = True
        return self.MECS.get_state(self.BSs), self.reward.reward, self.done, i
    
    def slot_step(self):
        shuffle(self.UEs)
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
        self.finish_reward = cn.finish_reward

    def task_failed(self):
        self.reward += self.fail_punish
        # raise Exception("only for debug")
        logging.info('任务失败后：reward is {}'.format(self.reward))

    def task_finish(self, task):
        self.reward += self.ce * task.energy + self.ct * task.time + \
                       self.finish_reward
        logging.debug('任务完成后：reward is {}'.format(self.reward))
        logging.debug('mode:{}, energy:{}, time:{}'.format(task.work_mode,
                      task.energy, task.time))
        del task
        
    def battery_empty(self):
        self.reward += self.c_battery_empty
        logging.info('电源没电后：reward is {}'.format(self.reward))
    
    def get_reward(self):
        return self.reward
    def reset(self):
        self.reward = 0
        # print('重置后：reward is {}'.format(self.reward))
        
        