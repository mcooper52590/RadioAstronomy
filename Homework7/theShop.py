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
<<<<<<< HEAD
plt.rc('xtick',labelsize=22)
plt.rc('ytick',labelsize=22)
=======
>>>>>>> 0025db2d98146882b05c7e0b3165705ee7382836

def sumOfSquares(dataActual, dataTheory):
    Sum = 0
    for i in range(dataTheory.shape[0]):
        if np.abs(dataActual[i] - dataTheory[i]):
            Sum = Sum + (dataActual[i] - dataTheory[i])**2
    return Sum

def getPhase_Theory(dBx, dBy, dBz, dec, ha):
   return (2*pi*nu/3e8)*(dBx*cos(dec)*cos(ha) - dBy*cos(dec)*sin(ha) + dBz*sin(dec))

<<<<<<< HEAD
def data_Even(data):
=======
def data_EvenPart(data):
>>>>>>> 0025db2d98146882b05c7e0b3165705ee7382836
    Len = int((data.shape[0]-1)/2)
    even = np.zeros([Len])
    for i in range(0, Len):
        even[i] = (1/2)*(data[Len+i] + data[Len-i])
    return even

<<<<<<< HEAD
def data_Odd(data):
=======
def data_OddPart(data):
>>>>>>> 0025db2d98146882b05c7e0b3165705ee7382836
    Len = int((data.shape[0]-1)/2)
    odd = np.zeros([Len])
    for i in range(0, Len):
        odd[i] = (1/2)*(data[Len+i] - data[Len-i])
    return odd

<<<<<<< HEAD
=======
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

>>>>>>> 0025db2d98146882b05c7e0b3165705ee7382836
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
<<<<<<< HEAD

def findMin_dBx_OnSpace(dBx, dBy, dBz, realEven, dec, ha):
    for i in range(dBx.shape[0]): 
        theory = getPhase_Theory(dBx[i], dBy, dBz, dec, ha)
        theoryEven = data_Even(theory)
        sqrs = sumOfSquares(realEven, theoryEven)
        if i == 0:
            minSqrs = sqrs
            minX = dBx[i]
        elif sqrs < minSqrs:
            minSqrs = sqrs
            minX = dBx[i] 
    return minX, minSqrs

def findMin_dBy_OnSpace(dBx, dBy, dBz, realOdd, dec, ha):
    for i in range(dBy.shape[0]): 
        theory = getPhase_Theory(dBx, dBy[i], dBz, dec, ha)
        theoryOdd = data_Odd(theory)
        sqrs = sumOfSquares(realOdd, theoryOdd)
        if i == 0:
            minSqrs = sqrs
            minY = dBy[i]
        elif sqrs < minSqrs:
            minSqrs = sqrs
            minY = dBy[i] 
    return minY, minSqrs

def findMin_dBz_OnSpace(dBx, dBy, dBz, realEven, dec, ha):
    for i in range(dBz.shape[0]): 
        theory = getPhase_Theory(dBx, dBy, dBz[i], dec, ha)
        theoryEven = data_Even(theory)
        sqrs = sumOfSquares(realEven, theoryEven)
        if i == 0:
            minSqrs = sqrs
            minZ = dBz[i]
        elif sqrs < minSqrs:
            minSqrs = sqrs
            minZ = dBz[i] 
    return minZ, minSqrs
#=======================================================================================================================
phaseArray = get_PhaseArray('PhaseError.txt')
ha = dtr(phaseArray[:,0]) 
dec = dtr(12.5)
nu = 5e9
for i in range(phaseArray.shape[0]):
    if phaseArray[i,0] == 0:
        dBz = (phaseArray[i,1]/(2*pi*nu/3e8)*sin(dec))

N = 20000
bnd = 2
dBy = 0    

dBx = np.linspace(-bnd, bnd, N)
realEven = data_Even(np.unwrap(dtr(phaseArray[:,1])))
dBx, dBxMinSqrs = findMin_dBx_OnSpace(dBx, dBy, dBz, realEven, dec, ha)

realOdd = data_Odd(np.unwrap(dtr(phaseArray[:,1])))
dBy = np.linspace(-bnd, bnd, N)
dBy, dByMinSqrs = findMin_dBy_OnSpace(dBx, dBy, dBz, realOdd, dec, ha)

#dBz = np.linspace(-bnd, bnd, N)
#sqrs12 = []
#sqrs33 = []
#for i in range(dBz.shape[0]):
#    theory12 = getPhase_Theory(dBx, dBy, dBz[i], dtr(12.5), ha)
#    theoryEven12 = data_Even(theory12)
#    realEven12 = data_Even(np.unwrap(dtr(phaseArray[:,1])))
#    sqrs12.append(sumOfSquares(realEven12%(2*pi), theoryEven12%(2*pi)))
#    
#    theory33 = getPhase_Theory(dBx, dBy, dBz[i], dtr(33.2), ha)
#    theoryEven33 = data_Even(theory33)
#    realEven33 = data_Even(np.unwrap(dtr(phaseArray[:,2])))
#    sqrs33.append(sumOfSquares(realEven33%(2*pi), theoryEven33%(2*pi)))
#

plt.clf() 



'''
Plotting the data versus theory even part for the 12.5 degree declination data
'''
#plt.plot(phaseArray[:120:2, 0], rtd(theoryEven%(2*pi)), color='red', label='Theory')
#realEven = data_Even(np.unwrap(dtr(phaseArray[:,1])))
#plt.scatter(phaseArray[:120:2, 0], rtd(realEven%(2*pi)), color='blue', label='Data')
#plt.legend(prop={'size': 28})
#plt.xlabel('Hour Angle (in degrees)', fontsize=28, labelpad=20)
#plt.ylabel('Phase (in degrees)', fontsize=28, labelpad=20)
#plt.legend(prop={'size': 28}, loc=[.3,.05])

'''
Plotting the data versus theory odd part for the 12.5 degree declination data
'''
#plt.plot(phaseArray[:120:2, 0], rtd(theoryOdd%(2*pi)), color='red', label='Theory')
#realOdd = data_Odd(np.unwrap(dtr(phaseArray[:,1])))
#plt.scatter(phaseArray[:120:2, 0], rtd(realOdd%(2*pi)), color='blue', label='Data')
#plt.legend(prop={'size': 28})
#plt.xlabel('Hour Angle (in degrees)', fontsize=28, labelpad=20)
#plt.ylabel('Phase (in degrees)', fontsize=28, labelpad=20)
#plt.legend(prop={'size': 28}, loc=[.3,.05])

'''
Plotting the data versus theory total for the 12.5 degree declination data
'''
#plt.plot(phaseArray[:, 0], rtd(theory%(2*pi)), color='red', label='Theory')
#real = dtr(phaseArray[:,1])
#plt.scatter(phaseArray[:, 0], rtd(real%(2*pi)), color='blue', label='Data')
#plt.legend(prop={'size': 28}, loc=[.3,.05])
#plt.xlabel('Hour Angle (in degrees)', fontsize=28, labelpad=20)
#plt.ylabel('Phase (in degrees)', fontsize=28, labelpad=20)

'''
Plotting the least squares for both declinations
'''
#plt.plot(dBz, sqrs12, color='blue', label='12.5 degree declination')
#plt.plot(dBz, sqrs33, color='green', label='33.2 degree declination')
#plt.legend(prop={'size': 28}, loc=[.05, 1.025], ncol=2)
#plt.xlabel('dBz', fontsize=28, labelpad=20)
#plt.ylabel('Least Squares Value', fontsize=28, labelpad=20)


'''
Plotting data versus theory total for both declinations
'''
#theory = getPhase_Theory(dBx, dBy, -1.565, dtr(12.5), ha)
#plt.plot(phaseArray[:, 0], rtd(theory%(2*pi)), color='red', label='Theory 12.5 Dec')
#real = dtr(phaseArray[:,1])
#plt.scatter(phaseArray[:, 0], rtd(real%(2*pi)), color='blue', label='Data 12.5 Dec')
#
#theory = getPhase_Theory(dBx, dBy, -1.565, dtr(33.2), ha)
#plt.plot(phaseArray[:, 0], rtd(theory%(2*pi)), color='orange', label='Theory 33.2 Dec')
#real = dtr(phaseArray[:,2])
#plt.scatter(phaseArray[:, 0], rtd(real%(2*pi)), color='green', label='Data 33.2 Dec')
#plt.legend(prop={'size': 24}, loc=[.2, 1.01], ncol=2)
#plt.xlabel('Hour Angle (in degrees)', fontsize=28, labelpad=20)
#plt.ylabel('Phase (in degrees)', fontsize=28, labelpad=20)


'''
Plotting data with theory subtracted with modulo
'''
theory = getPhase_Theory(dBx, dBy, -1.565, dtr(12.5), ha)
real = dtr(phaseArray[:,1]) - theory
plt.scatter(phaseArray[:, 0], rtd(real%(2*pi)), color='blue', label='12.5 Dec')

theory = getPhase_Theory(dBx, dBy, -1.565, dtr(33.2), ha)
real = dtr(phaseArray[:,2]) - theory
plt.scatter(phaseArray[:, 0], rtd(real%(2*pi)), color='green', label='33.2 Dec')
plt.legend(prop={'size': 24}, loc=[.2, 1.01], ncol=2)
plt.xlabel('Hour Angle (in degrees)', fontsize=28, labelpad=20)
plt.ylabel('Phase (in degrees)', fontsize=28, labelpad=20)

'''
Plotting data with theory subtracted without modulo
'''
#theory = getPhase_Theory(dBx, dBy, -1.565, dtr(12.5), ha)
#real = dtr(phaseArray[:,1]) - theory
#plt.scatter(phaseArray[:, 0], rtd(real), color='blue', label='12.5 Dec')
#
#theory = getPhase_Theory(dBx, dBy, -1.565, dtr(33.2), ha)
#real = dtr(phaseArray[:,2]) - theory
#plt.scatter(phaseArray[:, 0], rtd(real), color='green', label='33.2 Dec')
#plt.legend(prop={'size': 24}, loc=[.2, 1.01], ncol=2)
#plt.xlabel('Hour Angle (in degrees)', fontsize=28, labelpad=20)
#plt.ylabel('Phase (in degrees)', fontsize=28, labelpad=20)






















=======
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
>>>>>>> 0025db2d98146882b05c7e0b3165705ee7382836
