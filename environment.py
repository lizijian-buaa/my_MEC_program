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
import myrandom as ran

class MECsystem(object):
    def __init__(self, API_normalization=True, observer=None):
        # if API normalization is true, all inputs and outputs are resized to
        # [0,1], needed when using Neural Networks
        self.UEs = self.UserEquipments(cn.number, cn.X0)
        self.BSs = BSs(cn.BS2MECS_rate, cn.channel_gain, cn.width, cn.noise)
        observer.sync(self)
        self.reward = Reward(cn.coefficient_energy, cn.coefficient_time, \
                             cn.fail_punish, self, observer)
        self.MECS = server(need_preprocess=API_normalization)
        self.slot = cn.slot
        self.move_period = cn.move_period
        self.change_Prtask_period = cn.change_Prtask_period
        self.initialize()

        
    class UserEquipments(list):
        def __init__(self, num, X0):
            # prtask roughly be around ( 0.001, 0.1)
            self.PrUE = cn.PrUE
            self.change_Prtask = cn.change_Prtask
            self.X0 = X0 * self.change_Prtask["x0"]
            self.taskgroup = np.array([X0] * cn.task_group_num)
            self.num = num
            self.up = self.change_Prtask["up"] * self.X0
            self.down = self.change_Prtask["down"] * self.X0
            self.ins = self.change_Prtask["ins"]  # change towards mean value
            self.outs = self.change_Prtask["outs"]  # change away from mean value
            self.changeProbProb = cn.changeProbProb
            
        def showzone(self):
            # returns the location distribution over 4 zones
            table = np.zeros((1,4))
            for UE in self:
                table[0, UE.zone - 1] += 1
            # print(table)
            return table
        
        def move(self):
            for UE in self:
                UE.move(self.PrUE)
                
        def change_task_prob(self):
            for i in range(self.taskgroup.shape[0]):
                if ran.result(self.changeProbProb):
                    # increase prob.
                    pa = self.ins if self.taskgroup[i] < self.X0 else self.outs
                    self.taskgroup[i] = min(self.taskgroup[i]*pa, self.up)
                elif ran.result(self.changeProbProb/(1-self.changeProbProb)):
                    # decrease prob.
                    pa = self.ins if self.taskgroup[i] > self.X0 else self.outs
                    self.taskgroup[i] = max(self.taskgroup[i]/pa, self.down)
                    
    def initialize(self):
        self.done = False
        self.time = 0
        self.time_frozen = 0  # the attr to record time of the last decision
        self.clock = np.zeros(2)
        # clock of move period and Prtask change period
        for i in range(self.UEs.num):
            self.UEs.append(random_create_UE(self.UEs.taskgroup))
            # initialized UEs
        while not self.MECS.apply:
            self.time += self.slot
            self.slot_step()
        
    def step(self, action):
        self.time_frozen = self.time
        self.MECS.step(action, self.time, self.BSs, self.reward)
        logging.info('time is: {} of {}'.format(self.time, cn.time_total))
        self.MECS.apply = None
        while not self.MECS.apply:
            self.slot_step()
        if self.time > cn.time_total:
            self.done = True
        return self.MECS.get_state(self.BSs), self.reward.get_reward(), \
               self.done, self.time - self.time_frozen
    
    def slot_step(self):
        shuffle(self.UEs)
        self.time += self.slot
        self.clock += self.slot
        if self.clock[0] > self.move_period:
            self.clock[0] -= self.move_period
            self.UEs.move()
        if self.clock[1] > self.change_Prtask_period:
            self.clock[1] -= self.change_Prtask_period
            self.UEs.change_task_prob()
        for UE in self.UEs:
            UE.every_slot(self.time, self.slot, self.BSs, self.reward,
                          self.MECS)
     
    def reset(self, i=None, API_normalization=True, observer=None):
        # used as you want to test multiple conditions with a single run
        if i != None:
            number, X0 = cn.hypepairs[i]
            self.UEs = self.UserEquipments(number, X0)
        self.BSs = BSs(cn.BS2MECS_rate, cn.channel_gain, cn.width, cn.noise)
        if observer != None:
            observer.reset()
        self.reward = Reward(cn.coefficient_energy, cn.coefficient_time, \
                             cn.fail_punish, self, observer)
        self.MECS = server(need_preprocess=API_normalization)
        self.initialize()
        return self.MECS.get_state(self.BSs)
            
            
class Reward(object):
    def __init__(self, coefficient_energy, coefficient_time, fail_punish,
                 env, observer=None):
        self.reward = 0
        self.fail_punish = fail_punish
        self.ce = coefficient_energy
        self.ct = coefficient_time
        self.finish_reward = cn.finish_reward
        self.observer = observer
        self.env = env
        self.gamma = cn.gamma

    def task_failed(self):
        self.reward += self.fail_punish * pow(self.gamma,
                                        self.env.time-self.env.time_frozen)
        if self.observer != None:
            self.observer.fail()
        # raise Exception("only for debug")
        logging.info('任务失败后：reward is {}'.format(self.reward))

    def task_finish(self, task):
        self.reward += (self.ce * task.energy + self.ct * task.time + \
                       self.finish_reward) * pow(self.gamma,
                                         self.env.time-self.env.time_frozen)
        self.observer.finish_record(task)
        logging.debug('任务完成后：reward is {}'.format(self.reward))
        logging.debug('mode:{}, energy:{}, time:{}'.format(task.work_mode,
                      task.energy, task.time))
        del task
            
    def get_reward(self):
        self.observer.record_reward(self.reward)
        return self.reward
    
    def reset(self):
        self.reward = 0
        # print('重置后：reward is {}'.format(self.reward))
        
        
class Observer(object):
    def __init__(self):
        self.env = None
        self.delay = []
        self.energy = []
        self.count = 0
        self.task_num = 0
        self.offload_count = 0
        self.local_count = 0
        self.reward_history = []
        self.k = []
        
    def finish_record(self, task):
        self.delay.append(task.time)
        self.energy.append(task.energy)
        self.task_num += 1
        if task.work_mode == "local":
            self.local_count += 1
        elif task.work_mode == "offload":
            self.offload_count += 1
        
    def fail(self):
        self.task_num += 1
        self.count += 1
        
    def sync(self, env):
        # meant to get the time synchronized with the clock of env
        self.env = env
        
    def fail_rate(self):
        return self.count / self.task_num
    
    def record_reward(self, reward):
        self.reward_history.append(reward)
    
    def reset(self):
        self.delay = []
        self.energy = []
        self.count = 0
        self.task_num = 0
        self.offload_count = 0
        self.local_count = 0
        self.reward_history = []
        self.k = []
        
    def output_to_excel(self):
        # return list of data to be fulfilled in one row
        reward_mean = np.mean(self.reward_history)
        reward_var = np.std(self.reward_history)
        delay_mean = np.mean(self.delay)
        delay_var = np.std(self.delay)
        energy_mean = np.mean(self.energy)
        energy_var = np.std(self.energy)
        fail_rate = self.fail_rate()
        local_count = self.local_count
        offload_count = self.offload_count
        offloading_rate = self.offload_count / (self.local_count +
                                                self.offload_count)
        labels = [reward_mean, reward_var, delay_mean, delay_var, energy_mean,
          energy_var, fail_rate, offload_count, local_count, offloading_rate]
        return labels
    
