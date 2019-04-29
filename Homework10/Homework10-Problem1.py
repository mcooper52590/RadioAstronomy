# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

def dtr(d):
    return d*pi/180

def get_dx(x, spacing=.032):
    dx = np.zeros([x.shape[0]])
    dx[0] = (x[1] - x[0])/spacing
    dx[x.shape[0]-1] = (x[x.shape[0] - 1] - x[x.shape[0] - 2])/spacing
    for i in range(1, x.shape[0] - 2):
        dx[i] = (x[i+1] - x[i-1])/spacing
    return dx

def get_TempArray(file):
    f = open(file, 'r')
    incoming = f.read().split('\n')
    phaseArray = np.zeros([len(incoming), 3])
    for i, line in enumerate(incoming):
            holder = line.split(' ')
            holder2 = []
            for entry in holder:
                if entry:
                    holder2.append(float(entry))
            if holder2:
                phaseArray[i] = holder2
    phaseArray = phaseArray[:len(incoming) - 1,:]
    return phaseArray

tempArray = get_TempArray('/home/matthew/Main/RadioAstronomy/Homework10/Homework10Data')
plt.plot(tempArray[:,0], tempArray[:,1], color='black', label='Righthand Pol.')
plt.plot(tempArray[:,0], tempArray[:,2], color='black', linestyle='dashed', label='Lefthand Pol.')
plt.legend()
plt.yscale('log')
plt.xscale('log')
P = (tempArray[:,1] - tempArray[:,2])/(tempArray[:,1] + tempArray[:,2])

