#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 14:46:17 2021

@author: zijian
"""

#!/usr/bin/python
#coding=utf-8
# ==============================================================================
#
#       Filename:  demo.py
#    Description:  excel operat
#        Created:  Tue Apr 25 17:10:33 CST 2017
#         Author:  Yur
#
# ==============================================================================

import xlwt
import xlrd
from xlutils.copy import copy

rb = xlrd.open_workbook('result_data.xlsx')
workbook = copy(rb)
worksheet = workbook.get_sheet(0)

# 写入excel
# 参数对应 行, 列, 值
x=12
worksheet.write(70,0, label = x)

# 保存
workbook.save('result_data.xlsx')