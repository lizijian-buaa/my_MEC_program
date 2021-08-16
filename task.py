# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:50:08 2020

@author: lizijian
"""

import numpy as np

class Task():
    def __init__(self, data_size, computation_consumption, delay_tolerant,
                 UE, work_mode = None, arrival_timestamp=None):
        self.UE = UE
        self.data_size = data_size
        self.delay_tolerant = delay_tolerant
        self.computation_consumption = computation_consumption
        self.arrival_time = arrival_timestamp
        self.fail_time = self.arrival_time + self.delay_tolerant
        self.finish_time = None
        self.work_mode = work_mode
        self.time = None
        self.energy = None
        self.BS_index = None
        self.gain_index = None
        # self.trace = []
    
    def get_data_size(self):
        return self.data_size
    
    def get_arrival_time(self):
        return self.arrival_time
    
    def set_work_mode(self, work_mode):
        if work_mode in ('local', 'offload'):
            self.work_mode = work_mode
        else:
            raise Exception('unknown work mode {}, only local, offload is \
                            allowed'.format(work_mode))  
            # perhaps this print function won't work so neatly
        
    def set_offloading(self, BSs, BS, channel, MECS, MECS_f, now):
        # check weather the action is valid!!
        if MECS.f_free < MECS_f or BSs.busy[int(BS), int(channel)] or \
            self.UE.is_sending_occupied or not self.UE.available(int(BS)):
            # busy shall trigger a return
            
            return
        self.BS_index = BS
        self.gain_index = channel
        self.MECS_f = MECS_f
        self.finish_time = BSs.delay(self.UE, self, BS, channel) + \
                           MECS.delay(self) + now
        self.time = self.finish_time - self.arrival_time
        self.energy = BSs.energy(self.UE, self, BS, channel)
        self.UE.sending = self
        self.UE.is_sending_occupied = True
        # self.trace.append("buffer: start,{}".format(now-self.arrival_time))
        del self.UE.buffer[0] # delete task from waiting list
        
    def set_local(self, now):
        if self.UE.is_local_occupied:
            # check weather the action is valid!!
            return
        t = self.computation_consumption / self.UE.f
        self.finish_time = now + t
        self.time = self.finish_time - self.arrival_time
        self.energy = self.UE.P * t
        self.UE.local = self
        self.UE.is_local_occupied = True
        # self.trace.append("buffer: start,{}".format(now-self.arrival_time))
        del self.UE.buffer[0] # delete task from waiting list
        
    def start_work(self, now, reward, **kw):
        # ** key words used when offloading
        if self.work_mode == 'local':
            self.set_local(now)
        elif self.work_mode == 'offload':
            self.set_offloading(kw['BSs'], kw['BS'], kw['channel'], kw['MECS'],
                                kw['MECS_f'], now)
        if self.finish_time is None:
            # invalid action caused by Resources occupied
            return
        elif self.finish_time > self.fail_time:
# =============================================================================
#             this is a very important feature of this simulation model, which 
#             can omit the accumulation of tasks in buffers
# =============================================================================
            reward.task_failed()
            if self.work_mode == 'local':
                self.UE.local = None
                self.UE.is_local_occupied = False
            elif self.work_mode == 'offload':
                self.UE.sending = None
                self.UE.is_sending_occupied = False
                kw['BSs'].set_free(self)
                kw['MECS'].f_free += self.MECS_f
            
    def finish(self, BSs, MECS, reward):
        BSs.set_free(self)
        MECS.f_free += self.MECS_f
        reward.task_finish(self)
        # self.trace.append("self: finish")
      