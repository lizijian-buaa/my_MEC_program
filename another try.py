# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 21:55:40 2020

@author: lizijian
"""

import myrandom as ran
import matplotlib.pyplot as plt
import numpy as np

probunit = 0.001
pr_decrease = 0.01
pr_max = 0.01
probmax = 0.2

table = [0]*(round(probmax / probunit) + 1)  # for visualization
prob = 0

for i in range(100000):
    pr_increase = pr_max * (1 - prob / probmax)
    if ran.result(pr_increase):
        prob += probunit
    if ran.result(pr_decrease):
        prob -= probunit
        if prob < 0.1:
            prob = 0
    table[round(prob/probunit)] += 1

print(table)
# An "interface" to matplotlib.axes.Axes.hist() method
plt.bar(list(np.arange(0, probmax+probunit, probunit)), table)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('distribution Histogram')
plt.show()