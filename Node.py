# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 22:16:21 2020

@author: lizijian
"""

import uuid
from abc import abstractmethod, ABCMeta
import numpy as np

class Node(metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        self.uuid = uuid.uuid4()
        self.node_type = 0

    def print_me(self):
        pass
    
    def ran(self, Range):
        return np.random.uniform(Range[0], Range[1], 1)
        
    def ran01():
        return np.random.uniform(0, 1, 1)

    def get_uuid(self):
        return self.uuid.hex