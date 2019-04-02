#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:18:54 2019

@author: matthew
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pi = np.pi
cos=np.cos
sin=np.sin

def sumOfSquares(dataActual, dataTheory):
    Sum = 0
    for i in range(dataTheory.shape[0]):
        if np.abs(dataActual[i] - dataTheory[i]):
            Sum = Sum + (dataActual[i] - dataTheory[i])**2
    return Sum

def getPhase_Theory(dBx, dBy, dBz, dec, ha):
   return (2*pi*nu/3e8)*(dBx*cos(dec)*cos(ha) - dBy*cos(dec)*sin(ha) + dBz*sin(dec))

def data_EvenPart(data):
    Len = int((data.shape[0]-1)/2)
    even = np.zeros([Len])
    for i in range(0, Len):
        even[i] = (1/2)*(data[Len+i] + data[Len-i])
    return even

def data_OddPart(data):
    Len = int((data.shape[0]-1)/2)
    odd = np.zeros([Len])
    for i in range(0, Len):
        odd[i] = (1/2)*(data[Len+i] - data[Len-i])
    return odd

def findMin_dBx(N, bnd, dBy, dBz, phaseActual, ha, dec):
    phaseActual = phaseArray[:,1]
    phaseEven = data_EvenPart(phaseActual)
    dBx = np.linspace(-bnd, bnd, N)
    dBxMinVal = sumOfSquares(phaseEven%360, data_EvenPart(getPhase_Theory(dBx[0], dBy, dBz, dec, ha))%360)
    for i in range(1, dBx.shape[0]):
        hold = sumOfSquares(phaseEven%360, data_EvenPart(getPhase_Theory(dBx[i], dBy, dBz, dec, ha))%360)
        if  hold < dBxMinVal:
            print(dBxMinVal)
            dBxMinVal = hold
            dBxMin = dBx[i]
    return dBxMin

def findMin_dBy(N, bnd, dBx, dBz, phaseActual, ha, dec):
    phaseOdd = data_OddPart(phaseActual)
    dBy = np.linspace(-bnd, bnd, N)
    dBz = np.linspace(-bnd, bnd, N)
    dByMinVal = sumOfSquares(phaseOdd%360, data_OddPart(getPhase_Theory(dBx, dBy[0], dBz, dec, ha))%360)
    for i in range(1, dBy.shape[0]):
        hold = sumOfSquares(phaseOdd%360, data_OddPart(getPhase_Theory(dBx, dBy[i], dBz, dec, ha))%360)
        if  hold < dByMinVal:
            dByMinVal = hold
            dByMin = dBy[i]
    return dByMin

def get_PhaseArray(file):
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

def dtr(d):
    return d*pi/180

def rtd(r):
    return r/pi*180
#=======================================================================================================================
phaseArray = get_PhaseArray('PhaseError.txt')

evenReal = []
oddReal = []
for i in range(0, 60):
    evenReal.append((1/2)*(phaseArray[60+i,1] + phaseArray[60-i,1])%360)
    oddReal.append((1/2)*(phaseArray[60+i,1] - phaseArray[60-i,1])%360)
  
plt.plot(phaseArray[:120:2, 0], evenReal)    
#plt.plot(phaseArray[:120:2, 0], oddReal) 
  
dec = dtr(12.5)
nu = 5e9
ha = dtr(phaseArray[:,0])
for i in range(phaseArray.shape[0]):
    if phaseArray[i,0] == 0:
        dBz = (phaseArray[i,1]/(2*pi*nu/3e8)*sin(dec))
dBx = dBz

N = 100
bnd = 10
dBy = findMin_dBy(N, bnd, dBx, dBz, phaseArray[:,1], ha, dec)
dBx = findMin_dBx(N, bnd, dBy, dBz, phaseArray[:,1], ha, dec)

#dBy = findMin_dBy(N, bnd, dBx, dBz, phaseArray[:,1], ha, dec)
#dBx = findMin_dBx(N, bnd, dBy, dBz, phaseArray[:,1], ha, dec)

    
term1 = (2*pi*nu/3e8)*dBx*cos(dec)*cos(ha)
term2 = - (2*pi*nu/3e8)*dBy*cos(dec)*sin(ha)
term3 = (term2 + (2*pi*nu/3e8)*dBz*sin(dec) - term2)

pThe = (term1 + term2+ term3)%360
evenTheory = []
oddTheory = []
for i in range(0, 60):
    evenTheory.append((1/2)*(pThe[60+i] + pThe[60-i])%360)
    oddTheory.append((1/2)*(pThe[60+i] - pThe[60-i])%360) 
    
evenReal = np.array(evenReal)
evenTheory = np.array(evenTheory)

oddReal = np.array(oddReal)
oddTheory = np.array(oddTheory)
#evenCheck = data_EvenPart(pThe)
plt.plot(phaseArray[:120:2, 0], evenTheory) 
