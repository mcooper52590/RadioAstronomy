
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pi = np.pi
cos=np.cos
sin=np.sin

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

def sumOfSquares(theory, data):
    sumOSqrs = 0
    for i in range(theory.shape[0]):
        sumOSqrs = sumOSqrs + (data[i] - theory[i])**2
    return sumOSqrs

def dtr(d):
    return d*pi/180

def rtd(r):
    return r/pi*180

def minimize_Func(x, func, data, N, bnd):
    yGuess = np.linspace(-bnd, bnd, N)
    theory = func(x, yGuess)
    yMinVal = sumOfSquares(theory, data)
    for i in range(1, yGuess.shape[0]):
        theory = func(yGuess)
        hold = sumOfSquares_FromTheory(dBx[i], dBy, dBz, phaseArray, dec)
        if  hold < dBxMinVal:
            dBxMinVal = hold
            dBxMin = dBx[i]
    return dBxMin

def dBxFunc(ha, dBx, nu=5e9, dec=12.5):
    return (2*pi*nu/3e8)*(dBx*cos(dec)*cos(ha))
   
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

def get_dx(x, spacing=1):
    dx = np.zeros([x.shape[0]])
    dx[0] = (x[1] - x[0])/spacing
    dx[x.shape[0]-1] = (x[x.shape[0] - 1] - x[x.shape[0] - 2])/spacing
    for i in range(1, x.shape[0] - 2):
        dx[i] = (x[i+1] - x[i-1])/spacing
    return dx

def getPhase_Theory(dBx, dBy, dBz, dec, ha):
   return (2*pi*nu/3e8)*(dBx*cos(dec)*cos(ha) - dBy*cos(dec)*sin(ha) + dBz*sin(dec))

phaseArray = get_PhaseArray('PhaseError.txt')
unwrapPhs = np.unwrap(dtr(phaseArray[:,1]))
dec = dtr(12.5) 

evenReal = []
oddReal = []
for i in range(0, 60):
    evenReal.append((1/2)*(unwrapPhs[60+i] + unwrapPhs[60-i]))
    oddReal.append((1/2)*(unwrapPhs[60+i] - unwrapPhs[60-i]))
  
evenReal = np.array(data_EvenPart((phaseArray[:,1])))
oddReal = np.array(data_OddPart((phaseArray[:,1])))

N = 10000
bnd = 1
dBx = np.linspace(-bnd, bnd, N)          
  
dBx = -.1
dBy = 0
dBz = 0
#Section of code used to determine the error in the x-direction
#pThe = getPhase_Theory(dBx[i], dBy, dBz, dec, ha)
#evenTheory = data_EvenPart(pThe)
#dReal = get_dx(evenReal)
#dTh = get_dx(evenTheory)
#lstqMin = sumOfSquares(dTh, dReal)
#lstqTest = [lstqMin]
#dReal = get_dx(evenReal)
#for i in range(1, dBx.shape[0]):
#    pThe = getPhase_Theory(dBx[i], dBy, dBz, dec, ha)
#    evenTheory = data_EvenPart(pThe)
#    dTh = get_dx(evenTheory)
#    hold = sumOfSquares(dTh, dReal)
#    lstqTest.append(hold)
#    if hold < lstqMin:
#        dBxmin = dBx[i]
#        lstqMin = hold
#dBx = dBxmin  
pThe = getPhase_Theory(dBx, dBy, dBz, dec, ha)
plt.plot(phaseArray[:,0], pThe)
plt.plot(phaseArray[:,0], np.unwrap(dtr(phaseArray[:,1])))



  
#    oddTheory = []
#    for i in range(0, 60):
#        evenTheory.append((1/2)*(pThe[60+i] + pThe[60-i]))
#        oddTheory.append((1/2)*(pThe[60+i] - pThe[60-i])) 






#evenTheory = np.array(evenTheory)
dReal = get_dx(evenReal)
dTh = get_dx(evenTheory)
#plt.plot(ha[:120:2], dReal)   
#plt.plot(ha[:120:2], dTh) 
    
    
#plt.plot(phaseArray[:,0], np.unwrap(dtr(phaseArray[:,1])))
#plt.plot(ha[:120:2], evenReal)
#plt.plot(ha[:120:2], evenTheory)

#plt.plot(phaseArray[:,0], np.unwrap(phaseArray[:,2]))




#fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(17,12), dpi=166) 
#ax1.plot(ha[:120:2], evenReal)  
#ax1.plot(ha[:120:2], evenTheory)   
#ax2.plot(ha[:120:2], oddReal) 
#ax2.plot(ha[:120:2], oddTheory) 
#ax3.plot(ha, phaseArray[:,1])
#ax3.plot(ha, pThe)
#ax3.plot(ha, (phaseArray[:,1] - pThe), color='red')
#plt.plot(term2%360)
#dPhi = (2*pi*nu/3e8)*(dBx*cos(dec)*cos(ha) - dBy*cos(dec)*sin(ha) + dBz*sin(dec))

#fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False, 
#     figsize=(17,12), dpi=166) 
##
#ax1.scatter(phaseArray[:,0], phaseArray[:,1])
#ax1.plot()
#ax1.plot(phaseArray[:,0], (term1 + term2+ term3))



#
#effs1 = np.zeros([ha.shape[0], 2])
#effs1[:,0] = (2*pi*nu/3e8)*cos(dec1)*cos(ha)
#effs1[:,1] = -(2*pi*nu/3e8)*cos(dec1)*sin(ha)
#
#
#effs2 = np.zeros([ha.shape[0], 2]) 
#effs2[:,0] = (2*pi*nu/3e8)*cos(dec2)*cos(ha)
#effs2[:,1] = -(2*pi*nu/3e8)*cos(dec2)*sin(ha)
#
#solnArr = np.zeros([ha.shape[0], 2]) 
#for i in range(ha.shape[0]):
#    coeffArr = [effs1[i,:], effs2[i,:]]
#    resArr = [phaseArray[i,1] - dBz*(2*pi*nu/3e8)*sin(dec1), 
#              phaseArray[i,2] - dBz*(2*pi*nu/3e8)*sin(dec2)]
#    solnArr[i] = np.linalg.lstsq(coeffArr, resArr)[0]





#phaseDiffs = (1/2)*(phaseArray[:,2] - phaseArray[:,1])%360
#effs3 = np.zeros([ha.shape[0], 3]) 
#effs3[:,0] = effs2[:,0] - effs1[:,0]
#effs3[:,1] = effs2[:,1] - effs1[:,1]
#effs3[:,2] = effs2[:,2] - effs1[:,2]
#

#    
#solnArrDiff[np.where(np.abs(solnArrDiff) > 10)] = 0
#
#
#phaseSums = (1/2)*(phaseArray[:,2] + phaseArray[:,1])%360
#effs3 = np.zeros([ha.shape[0], 3]) 
#effs3[:,0] = effs2[:,0] + effs1[:,0]
#effs3[:,1] = effs2[:,1] + effs1[:,1]
#effs3[:,2] = effs2[:,2] + effs1[:,2]
#
#solnArrSum = np.zeros([ha.shape[0], 3]) 
#for i in range(ha.shape[0]):
#    coeffArr = [effs1[i,:], effs2[i,:], effs3[i,:]]
#    resArr = [phaseArray[i,1], phaseArray[i,2], phaseSums[i]]
#    solnArrSum[i] = np.linalg.lstsq(coeffArr, resArr)[0]
#    
#solnArrSum[np.where(np.abs(solnArrSum) > 10)] = 0






#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#N = 5
#step = 1000
#dBx = np.linspace(-N, N, step)
#dBy = np.linspace(-N, N, step)
#
#
#DBx, DBy = np.meshgrid(dBx, dBy)
#pArrTheory = np.zeros([dBx.shape[0], dBy.shape[0]])
#dBzTerm = (2*pi*nu/3e8)*dBz*sin(dec)
#for i in range(dBx.shape[0]):
#    for j in range(dBx.shape[0]):
#        term1 = (2*pi*nu/3e8)*dBx[i]*cos(dec)*cos(ha[0])
#        term2 = - (2*pi*nu/3e8)*dBy[j]*cos(dec)*sin(ha[0])
#        term3 = term1 + term2 + dBzTerm
#        pArrTheory[i,j] = term3
#       
#ax.plot_surface(DBx, DBy, pArrTheory)        
#for i in range(dBx.shape[0]):
#    term1 = (2*pi*nu/3e8)*dBx[i]*cos(dec)*cos(ha[0])
#    for j in range(dBy.shape[0]):
#        term2 = - (2*pi*nu/3e8)*dBy[j]*cos(dec)*sin(ha[0])
#        term3 = term2 + (2*pi*nu/3e8)*dBz*sin(dec) - term2
#        pArrTheory[i,j] = term3





#ax1.plot(phaseArray[:,0], term3)
#ax1.scatter(phaseArray[:,0], term3)

#ax2.scatter(phaseArray[:,0], phaseArray[:,2])    
      