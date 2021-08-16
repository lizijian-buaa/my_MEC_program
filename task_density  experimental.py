# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 17:36:30 2020

@author: lizijian
"""



import myrandom as ran
import matplotlib.pyplot as plt

group_num = 3
prmin = 0.001
momentum = 0.98
pr_change = 0.05

prob = 0
probset = []

for i in range(1000000):
    if i % 10 is 0:
        prob *= momentum
    if ran.result(pr_change):
        prob += prmin
    if i % 100 is 0:
        probset.append(prob)
# plt.plot(probset)
plt.hist(probset)