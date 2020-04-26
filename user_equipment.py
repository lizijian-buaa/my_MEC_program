# -*- coing: utf-8 -*-
"""
Created on Fri Mar 13 13:39:31 2020

中期初步程序伪码:
UEs=[]
initialize:
    reward
    MECS
    BSs
    Agent
for i in range(20):
    UEs.append(random_create_UE())
for(time=0,i=0; time+=slot; time<total_time):
    i++
    for UE in UEs:
        UE.set_battery()
        UE.set_charge()
        if UE.b is 0:
            if battery just empty:
                reward.battery_empty()
                if UE.sending is not None or UE.local is not None
                    set UE.sending and/or UE.local fail and call reward.fail() \
                    accordingly
        else:
            random create task with certain probability
            UE.buffer.append(new_task)
            if UE.buffer is not []:
                MECS.apply(UE.buffer[0]) -- add to apply list
            check the completion of UE.sending and UE.local and delete the \
            completed one(s) then set reward accordingly    
    if len(MECS.apply) > the input UE num of MECSnet(denoted by numUE):
        for all apply determine its OffloadingDemand(OD) via UEnet
        del the apply other than the top numUE OD ones, set the delected work \
        mode as 'local'
        MECS.apply()
    while len(MECS.apply) < numUE:
        MECS.apply.add_virtual_task()
    observe state
    get action, reward given state
    memory (old_state, action, state, reward)
    reward.reset()
    train MECSnet            
    MECS.apply = []
    for UE in UEs:
        if UE.buffer is not []:
            UE.buffer[0].start_work() -- set finish time, energy consumption \
            etc
            
@author: lizijian
"""
from constants import *
import numpy as np
import logging
from Node import Node
from task import Task
from myrandom import ran, ran01, result

class UserEquipment(Node):
    def __init__(self, full_Battery, CPU_frequency, power_density, P_standby, \
                 P_send, Prtask, charging_speed_range, x1, y1, x2, y2, \
                 battery = None):
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
        self.Prtask = Prtask
        self.ischarging = False
        self.charging_speed_range = charging_speed_range 
        self.buffer = []
        # this should be a range
        self.charging_speed = ran(charging_speed_range)
        self.is_local_occupied = False
        self.local = None
        self.is_sending_occupied = False
        self.sending = None
        # not only during sending time, but also its execution by MECS
        self.just_empty = True
        
    def set_charge(self):
        if self.ischarging:
            if result(max(self.x2 * self.b / self.B + self.y2, 0)):
                self.ischarging = False
        else:
            if result(max(self.x1 * self.b / self.B + self.y1, 0)):
                self.ischarging = True
                self.charging_speed = ran(self.charging_speed_range)
            
    def set_battery(self, slot):
        rate = - self.P_standby
        if self.ischarging:
            rate += self.charging_speed
        if self.is_local_occupied:
            rate -= self.P
        if self.is_sending_occupied:
            rate -= self.P_send
        self.b += rate * slot
        if self.b > self.B:
            self.b = self.B
        if self.b < 0:
            self.b = 0
            
    def every_slot(self, time, slot, BSs, reward, MECS):
        # self.set_battery(slot)
        # self.set_charge()
        self.set_tasks(time, BSs, reward, MECS)
        
    def set_tasks(self, time, BSs, reward, MECS):        
        # if battery out of power        
        if self.b == 0:
            if self.just_empty:
                reward.battery_empty()
                self.just_empty = False
            if self.sending is not None:
                reward.task_failed()
                self.sending = None  # not sure is this completely delected
            if self.is_local_occupied:
                reward.task_failed()
                self.local = None
        
        else:
            self.just_empty = True
            # receive random new task
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
                if self.sending.finish_time > time:
                    self.is_sending_occupied = False
                    self.sending.finish(BSs, MECS, reward)
                    # free basestation occupation and set reward
                    self.sending = None
            if self.is_local_occupied:
                if self.local.finish_time > time:
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