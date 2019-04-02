
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



def dtr(d):
    return d*pi/180

def rtd(r):
    return r/pi*180

#
#def findMin_dBy(N, bnd, dBx, dBz, phaseArray, dec):
#    dBy = np.linspace(-bnd, bnd, N)
#    dByMinVal = sumOfSquares_FromTheory(dBx, dBy[0], dBz, phaseArray, dec)
#    for i in range(1, dBy.shape[0]):
#        hold = sumOfSquares_FromTheory(dBx, dBy[i], dBz, phaseArray, dec)
#        if  hold < dByMinVal:
#            dByMinVal = hold
#            dByMin = dBy[i]
#    return dByMin
#
#def findMin_dBz(N, bnd, dBx, dBy, phaseArray, dec):
#    dBz = np.linspace(-bnd, bnd, N)
#    dBzMinVal = sumOfSquares_FromTheory(dBx, dBy, dBz[0], phaseArray, dec)
#    for i in range(1, dBz.shape[0]):
#        hold = sumOfSquares_FromTheory(dBx, dBy, dBz[i], phaseArray, dec)
#        if  hold < dBzMinVal:
#            dBzMinVal = hold
#            dBzMin = dBz[i]
#    return dBzMin


N = 1000
bnd = 10

#dBy = np.linspace(-bnd, bnd, N)
#dBz = np.linspace(-bnd, bnd, N)

#dBy = dBz
#dBxHold = dBz
#dByHold = dBy
#dBzHold = dBz
#
#error = 1e-3
#findX = True
#findY = True
#findZ = True
#i = 0
#while findX==True and findY==True and findZ==True:
#    if findX==True:
#        dBx = findMin_dBx(N, bnd, dBy, dBz, phaseArray, dtr(12.5))
#    if findY==True:
#        dBy = findMin_dBy(N, bnd, dBx, dBz, phaseArray, dtr(12.5))
#    if findZ ==True:
#        dBz = findMin_dBz(N, bnd, dBx, dBy, phaseArray, dtr(12.5))
#    if np.abs(dBx - dBxHold) < error:
#        findX = False
#    if np.abs(dBy - dByHold) < error:
#        findY = False
#    if np.abs(dBz - dBzHold) < error:
#        findZ = False
#    dBxHold = dBx
#    dByHold = dBy
#    dBzHold = dBz
#    print(i)
#    i += 1


#dBx = 
#dByTest = np.zeros([dBy.shape[0]])
#dByMin = sumOfSquares_FromTheory(dBx, dBy[0], dBz, phaseArray, dtr(12.5))
#dByMinIn = 0
#for i in range(1, dBx.shape[0]):
#    dBxTest[i] = sumOfSquares_FromTheory(dBx, dBy[i], dBz, phaseArray, dtr(12.5))
#    if dBxTest[i] < dBxMin:
#        dBxMin = dBxTest[i]
#        dBxIn = i
##        
#        
#dBxTest = np.zeros([dBx.shape[0]])
#dBxMin = sumOfSquares_FromTheory(dBx[0], dBy, dBz, phaseArray, dtr(12.5))
#dBxMinIn = 0
#for i in range(1, dBx.shape[0]):
#    dBxTest[i] = sumOfSquares_FromTheory(dBx, dBy, dBz[i], phaseArray, dtr(12.5))
#    if dBxTest[i] < dBxMin:
#        dBxMin = dBxTest[i]
#        dBxIn = i   


def sumOfSquares(dataActual, dataTheory):
    Sum = 0
    for i in range(dataTheory.shape[0]):
        Sum = Sum + (dataActual[i] - dataTheory[i])**2
    return Sum

#sos = sumOfSquares(evenReal, evenTheory)

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
        odd[i] = (1/2)*(data[Len+i] + data[Len-i])
    return odd



def findMin_dBy(N, bnd, dBx, dBz, phaseActual, ha, dec):
    phaseOdd = data_OddPart(phaseActual)
    dBy = np.linspace(-bnd, bnd, N)
    dByMinVal = sumOfSquares(phaseOdd%360, data_OddPart(getPhase_Theory(dBx, dBy[0], dBz, dec, ha))%360)
    for i in range(1, dBx.shape[0]):
        hold = sumOfSquares(phaseOdd%360, data_OddPart(getPhase_Theory(dBx, dBy[i], dBz, dec, ha))%360)
        if  hold < dByMinVal:
            dByMinVal = hold
            dByMin = dBy[i]
    return dByMin
#
phaseArray = get_PhaseArray('PhaseError.txt')
ha = dtr(phaseArray[:,0])
evenReal = []
oddReal = []
for i in range(0, 60):
    evenReal.append((1/2)*(phaseArray[60+i,1] + phaseArray[60-i,1])%360)
    oddReal.append((1/2)*(phaseArray[60+i,1] - phaseArray[60-i,1])%360)
  
#plt.plot(phaseArray[:120:2, 0], evenReal)    
#plt.plot(phaseArray[:120:2, 0], oddReal) 
  
dec = dtr(12.5)
nu = 5e9

for i in range(phaseArray.shape[0]):
    if phaseArray[i,0] == 0:
        dBz = (phaseArray[i,1]/(2*pi*nu/3e8)*sin(dec))
dBy = 2
N = 1000
bnd = 10
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


dBx
#dBx = findMin_dBx(100, 10, dBy, dBz, phaseArray[:,1], ha, dec)  
#dBy = findMin_dBx(100, 10, dBx, dBz, phaseArray[:,1], ha, dec) 
#dBx = findMin_dBx(100, 10, dBy, dBz, phaseArray[:,1], ha, dec) 
#dBx = findMin_dBx(100, 10, dBy, dBz, phaseArray[:,1], ha, dec) 

term1 = (2*pi*nu/3e8)*dBx*cos(dec)*cos(ha)
term2 = - (2*pi*nu/3e8)*dBy*cos(dec)*sin(ha)
term3 = (term2 + (2*pi*nu/3e8)*dBz*sin(dec) - term2)

plt.plot(ha, (term1 + term2+ term3)%360)
pThe = (term1 + term2+ term3)%360
evenTheory = []
oddTheory = []
for i in range(0, 60):
    evenTheory.append((1/2)*(pThe[60+i] + pThe[60-i])%360)
    oddTheory.append((1/2)*(pThe[60+i] - pThe[60-i])%360) 
    
evenReal = np.array(evenReal)
evenTheory = np.array(evenTheory)

evenCheck = data_EvenPart(pThe)
plt.plot(phaseArray[:120:2, 0], evenTheory) 


 
#plt.plot(ha[:120:2], oddTheory) 
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
      