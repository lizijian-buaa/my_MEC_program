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
    
    def set_OD(self, UEnet):
        # get local state x
        x = [self.data_size, self.computation_consumption, self.UE.b,
             self.UE.B, self.UE.f, self.UE.energy_per_cycle, self.UE.P_send,
             self.UE.ischarging]
        self.od = UEnet.choose_action(np.array(x).astype(float))
        
    def set_offloading(self, BSs, BS, channel, MECS, MECS_f, now):
        # check weather the action is valid!!
        if MECS.f_free < MECS_f or BSs.busy[int(BS), int(channel)] or \
            self.UE.is_sending_occupied:
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
        del self.UE.buffer[0] # delete task from waiting list
        
    def set_local(self, now):
        # check weather the action is valid!!
        if self.UE.is_local_occupied:
            return
        t = self.computation_consumption / self.UE.f
        self.finish_time = now + t
        self.time = self.finish_time - self.arrival_time
        self.energy = self.UE.P * t
        self.UE.local = self
        self.UE.is_local_occupied = True
        del self.UE.buffer[0] # delete task from waiting list
        
    def start_work(self, now, reward, **kw):
        if self.work_mode is 'local':
            self.set_local(now)
        elif self.work_mode is 'offload':
            self.set_offloading(kw['BSs'], kw['BS'], kw['channel'], kw['MECS'],
                                kw['MECS_f'], now)
        if self.finish_time > self.fail_time:
            reward.task_failed()
            del self
            
    def finish(self, BSs, MECS, reward):
        BSs.set_free(self)
        MECS.f_free += self.MECS_f
        reward.task_finish(self)
        del self
      