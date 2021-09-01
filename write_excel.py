#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 16:11:23 2021

@author: zijian
"""
import xlwt
import xlrd
from xlutils.copy import copy


class Write_result():
    # write a group of result data that generated with the same useMECS and
    # decision method to excel cells
    def __init__(self, file_path, sheet_index=0):
        self.file_path = file_path
        rb = xlrd.open_workbook(self.file_path)
        self.workbook = copy(rb)  # the file to be manipulated with
        self.sheet = self.workbook.get_sheet(sheet_index)
        self.head = ["number, x0", "reward_mean", "reward_var", "delay_mean", 
                     "delay_var", "energy_mean", "energy_var", "fail_rate",
                     "offload_count", "local_count", "offload_rate"]
        self.row = 0  # currently writing in this row
    
    def write_head(self, method):
        for i in range(len(self.head)):
            self.sheet.write(self.row, i, label=self.head[i])
        self.sheet.write(self.row, len(self.head)+1, label=method)
        self.row += 1
    
    def write_data(self, data):
        # write a row of data
        for i in range(len(data)):
            self.sheet.write(self.row, i, label = data[i])
        self.row += 1
    
    def save(self):
        self.workbook.save(self.file_path)
        
        