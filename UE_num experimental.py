# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:20:48 2020

@author: lizijian
"""

import numpy as np
import myrandom as ran
import matplotlib.pyplot as plt

'''
# square zone

PrUE = np.array([[None, 0.0002, 0.0001, 0.0002, 0.0004],
                 [0.0002, None, 0.0002, 0.0001, 0.0004],
                 [0.0001, 0.0002, None, 0.0002, 0.0004],
                 [0.0002, 0.0001, 0.0002, None, 0.0004],
                 [0.0004, 0.0004, 0.0004, 0.0004, None]])

PrUE = [PrUE if element is not None else PrUE]
'''

# triangle zone
PrUE = np.array([[None, 0.001, 0.001, 0.002],
                 [0.001, None, 0.001, 0.002],
                 [0.001, 0.001, None, 0.002],
                 [0.002, 0.002, 0.002, None]])


N = 30
n = round(N/2)
pr_max = 0.1
var = 0.2
table = [0]*(N + 1)
for i in range(1000000):
    pr_increase = pr_max * (1 - var*n/N)
    pr_decrease = pr_max * (1 - var + var*n/N)
    if ran.result(pr_increase):
        n += 1
        n = min(N,n)
    if ran.result(pr_decrease):
        n -= 1
        n = max(n,0)
    table[n] += 1

print(table)
# An "interface" to matplotlib.axes.Axes.hist() method
plt.bar(range(N+1), table)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('distribution Histogram')
plt.show()


# Transfer Probability Matrix of UE per seconds

'''
import myrandom as ran
import matplotlib.pyplot as plt

N = 30
n = round(N/2)
pr_max = 0.1
var = 0.2
table = [0]*(N + 1)
for i in range(1000000):
    pr_increase = pr_max * (1 - var*n/N)
    pr_decrease = pr_max * (1 - var + var*n/N)
    if ran.result(pr_increase):
        n += 1
        n = min(N,n)
    if ran.result(pr_decrease):
        n -= 1
        n = max(n,0)
    table[n] += 1

print(table)
# An "interface" to matplotlib.axes.Axes.hist() method
plt.bar(range(N+1), table)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('distribution Histogram')
plt.show()
'''