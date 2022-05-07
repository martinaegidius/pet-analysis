#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 09:33:04 2022

@author: max
"""

from scipy import stats
import matplotlib.pyplot as plt
from systemMatrix import * 

shapiro = []

data = np.delete(data,0,axis=2)

for k in range(data.shape[2]):
    for i in range(data.shape[1]):
        x = []
        for j in range(data.shape[0]):
            x.append(data[j,i,k]) #generate corresponding samples 
        #print("testing shapiro on" + str(x))
        shapiro.append(stats.shapiro(x)[1])

plt.plot(range(len(shapiro)),shapiro)
plt.ylim((0,0.10))

count = [i for i in shapiro if i <= 0.05]