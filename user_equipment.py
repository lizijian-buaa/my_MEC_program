# -*- coing: utf-8 -*-
"""
Created on Fri Mar 13 13:39:31 2020
@author: lizijian
"""
import constants as cn
import numpy as np
from task import Task
from myrandom import ran, ran01, result

class UserEquipment():
    def __init__(self, task_group, CPU_frequency, power_density, P_send,
                 task_group_index, zone = np.random.randint(1,4)):
        self.f = CPU_frequency
        self.z = power_density
        self.energy_per_cycle = self.z * self.f
        self.P = self.energy_per_cycle * self.f
        self.P_send = P_send
        self.buffer = []
        self.is_local_occupied = False
        self.local = None
        self.is_sending_occupied = False
        self.sending = None
        # not only during sending time, but also its execution by MECS
        self.zone = zone  # Category ID number
        self.next_zone = None
        self.task_group = task_group
        self.task_group_index = task_group_index # from 0 to task_group_num-1
        # defect: cannot reveal the periodic change of Env (i.e. daily)
            
    def every_slot(self, time, slot, BSs, reward, MECS):
        self.generate_tasks(time, reward)
        self.set_tasks(time, BSs, reward, MECS)
        self.check_task_finish(time, BSs, reward, MECS)
        
    def generate_tasks(self, time, reward):
        if result(self.task_group[self.task_group_index]):
            new_task = self.random_create_task(time)
            self.buffer.append(new_task)
        for i, task in enumerate(self.buffer):
            if task.fail_time < time:
                # task generated before "delay_tolerant" amount of time ago
                self.buffer.pop(i)
                reward.task_failed()
            else: 
                break
        
    def check_task_finish(self, time, BSs, reward, MECS):
        # check tasks finishment
        if self.is_local_occupied:
            if self.local.finish_time < time:
                self.is_local_occupied = False
                reward.task_finish(self.local)
                self.local = None
        if self.is_sending_occupied:
            if self.sending.finish_time < time:
                self.is_sending_occupied = False
                self.sending.finish(BSs, MECS, reward)
                # the "finish" method frees basestation occupation and set
                # reward
                self.sending = None
        
    def set_tasks(self, time, BSs, reward, MECS):
        if self.buffer:
            task = self.buffer[0]
            if self.zone == 4:
                # if zone 4, automatically set task_workmode local
                if not self.is_local_occupied:
                    task.set_work_mode('local')
                    task.start_work(time, reward)
            else:
                # not zone 4
                MECS.offloading_apply(task)
                                        
    def random_create_task(self, time):
        return Task(ran(cn.data_size), ran(cn.computation_consumption),
                    cn.delay_tolerant, self, arrival_timestamp = time)
    
    def move(self, PrUE):
        if not self.next_zone:
            if self.is_sending_occupied:
                self.next_zone = 1 + np.random.choice([i for i in \
                                range(PrUE.shape[0])], 1,
                                p = list(PrUE[self.zone - 1]))[0]
            else:
                original_zone = self.zone
                self.zone = 1 + np.random.choice([i for i in \
                            range(PrUE.shape[0])], 1,
                            p = list(PrUE[self.zone - 1]))[0]
                if self.zone == 4 and original_zone != 4:
                    self.reset_parameters()
        elif not self.is_sending_occupied:
                self.zone = self.next_zone
                self.next_zone = None
                
    def available(self, BS):
        if BS == 0 or self.zone == BS:     
            return True
        else:
            return False
                
    def reset_parameters(self):
        self.f = ran(cn.UE_frequency)
        self.z = ran(cn.power_density)
        self.energy_per_cycle = self.z * self.f
        self.P = self.energy_per_cycle * self.f
        self.P_send = ran(cn.P_send)
        self.task_group_index = np.random.randint(cn.task_group_num)
    
    
def random_create_UE(taskgroup):
    return UserEquipment(taskgroup, ran(cn.UE_frequency),
                         ran(cn.power_density), ran(cn.P_send),
                         np.random.randint(cn.task_group_num))

