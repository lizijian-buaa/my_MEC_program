# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:45:30 2020

@author: lizijian
"""

import numpy as np

def ran(Range):
    return np.random.uniform(Range[0],Range[1])

def ran01():
    return np.random.uniform(0, 1)

def result(prob):
    if 0 < prob < 1:
        return ran01() < prob
    else:
        return None
