# -*- coing: utf-8 -*-
"""
Created on Fri Mar 13 13:39:31 2020
@author: lizijian
"""
from constants import *
import numpy as np
from task import Task
from myrandom import ran, ran01, result

class UserEquipment():
    def __init__(self, full_Battery, CPU_frequency, power_density, P_standby, \
                 P_send, charging_speed_range, x1, y1, x2, y2, zone, \
                 task_group ,battery = None):
        super().__init__()
        self.B = full_Battery
        self.b = battery
        if self.b == None:
            self.b = self.B
        self.f = CPU_frequency
        self.z = power_density
        self.energy_per_cycle = self.z * self.f
        self.P = self.energy_per_cycle * self.f
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.P_standby =  P_standby
        self.P_send = P_send
#       self.Prtask = Prtask
        self.buffer = []
        self.is_local_occupied = False
        self.local = None
        self.is_sending_occupied = False
        self.sending = None
        # not only during sending time, but also its execution by MECS
        self.zone = zone  # Category ID number
        self.task_group = task_group # ，设几个（3）group的UE，每组UE任务到来密度一致。
        # 本设置的defect，不能反映出程周期规律变化的环境
            
    def every_slot(self, time, slot, BSs, reward, MECS):
        self.set_tasks(time, BSs, reward, MECS)
        
    def set_tasks(self, time, BSs, reward, MECS):        
        if result(self.Prtask):
            new_task = self.random_create_task(time)
            self.buffer.append(new_task)
        while self.buffer != []:
            if self.buffer[0].fail_time < time:
                reward.task_failed()
                del self.buffer[0]
            else:
                MECS.offloading_apply(self.buffer[0])
            break
                
        # check tasks finishment
        if self.is_sending_occupied:
            if self.sending.finish_time < time:
                self.is_sending_occupied = False
                self.sending.finish(BSs, MECS, reward)
                # free basestation occupation and set reward
                self.sending = None
        if self.is_local_occupied:
            if self.local.finish_time < time:
                self.is_local_occupied = False
                reward.task_finish(self.local)       
                self.local = None
                    
    def random_create_task(self, time):
        return Task(ran(data_size), ran(computation_consumption),
                    delay_tolerant, self, arrival_timestamp = time)
            
def random_create_UE():
    return UserEquipment(ran(full_Battery), ran(UE_frequency),
                         ran(power_density), ran(P_standby), ran(P_send),
                         Prtask, charging_speed_range, ran(x1), ran(y1),
                         ran(x2), ran(y2))