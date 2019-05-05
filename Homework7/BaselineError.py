<<<<<<< HEAD
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:18:54 2019

@author: matthew
"""

=======

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
>>>>>>> 0025db2d98146882b05c7e0b3165705ee7382836
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pi = np.pi
cos=np.cos
sin=np.sin
<<<<<<< HEAD
plt.rc('xtick',labelsize=22)
plt.rc('ytick',labelsize=22)

def sumOfSquares(dataActual, dataTheory):
    '''
    Calculates the sum of the distances from the theory to the data values 
    '''
    Sum = 0
    for i in range(dataTheory.shape[0]):
        if np.abs(dataActual[i] - dataTheory[i]):
            Sum = Sum + (dataActual[i] - dataTheory[i])**2
    return Sum

def getPhase_Theory(dBx, dBy, dBz, dec, ha, nu):
    '''
    Uses the equation (3) given in Lecture 7 to calculate the theoretical phase shift based on 
    the error in baseline measurements.
    '''
   return (2*pi*nu/3e8)*(dBx*cos(dec)*cos(ha) - dBy*cos(dec)*sin(ha) + dBz*sin(dec))

def data_Even(data):
    '''
    Calculates even part of a set of data.  Return array is half the size.  Currently rigged for odd arrays.
    Evenly-sized array won't work properly.
    '''
    Len = int((data.shape[0]-1)/2)
    even = np.zeros([Len])
    for i in range(0, Len):
        even[i] = (1/2)*(data[Len+i] + data[Len-i])
    return even

def data_Odd(data):
    '''
    Calculates odd part of a set of data.  Return array is half the size.  Currently rigged for odd arrays.
    Evenly-sized array won't work properly.
    '''
    Len = int((data.shape[0]-1)/2)
    odd = np.zeros([Len])
    for i in range(0, Len):
        odd[i] = (1/2)*(data[Len+i] - data[Len-i])
    return odd

def get_PhaseArray(file):
    '''
    Takes the three column input file and returns a numpy array with the values from the text file
    '''
=======

def get_PhaseArray(file):
>>>>>>> 0025db2d98146882b05c7e0b3165705ee7382836
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

<<<<<<< HEAD
def dtr(d):
    '''
    Degrees to radians
    '''
    return d*pi/180

def rtd(r):
    '''
    Radians to degrees
    '''
    return r/pi*180

def findMin_dBx_OnSpace(dBx, dBy, dBz, realEven, dec, ha, nu):
    '''
        dBx = a range of possible values for dBx. Values should be in a [-a, a] format,
              since this relys heavily on the convexity of the theory for this particular 
              function
        dBy = a float value for the dBy guess
        dBz = a float value for the dBz guess
        realEven = even part of the data
        dec =  phase center declination
        ha = pitch angle sample points (in degrees)
        nu = frequency at which the values were sampled
    '''
    for i in range(dBx.shape[0]): 
        theory = getPhase_Theory(dBx[i], dBy, dBz, dec, ha, nu)
        theoryEven = data_Even(theory)
        sqrs = sumOfSquares(realEven, theoryEven)
        if i == 0:
            minSqrs = sqrs
            minX = dBx[i]
        elif sqrs < minSqrs:
            minSqrs = sqrs
            minX = dBx[i] 
    return minX, minSqrs

def findMin_dBy_OnSpace(dBx, dBy, dBz, realOdd, dec, ha, nu):
        '''
        dBy = a range of possible values for dBx. Values should be in a [-a, a] format,
              since this relys heavily on the convexity of the theory for this particular 
              function
        dBx = a float value for the dBx guess
        dBz = a float value for the dBz guess
        realEven = even part of the data
        dec =  phase center declination
        ha = pitch angle sample points (in degrees)
    '''
    for i in range(dBy.shape[0]): 
        theory = getPhase_Theory(dBx, dBy[i], dBz, dec, ha, nu)
        theoryOdd = data_Odd(theory)
        sqrs = sumOfSquares(realOdd, theoryOdd)
        if i == 0:
            minSqrs = sqrs
            minY = dBy[i]
        elif sqrs < minSqrs:
            minSqrs = sqrs
            minY = dBy[i] 
    return minY, minSqrs

#=======================================================================================================================
'''
Importing the phase Array, setting variables for different functions
'''
phaseArray = get_PhaseArray('PhaseError.txt')
ha = dtr(phaseArray[:,0]) 
dec = dtr(12.5)
nu = 5e9

'''
Getting initial guess for dBz
'''
for i in range(phaseArray.shape[0]):
    if phaseArray[i,0] == 0:
        dBz = (phaseArray[i,1]/(2*pi*nu/3e8)*sin(dec))

'''
Setting initial boundaries and the sampling density for dBz 
'''
N = 20000
bnd = 2
dBy = 0    

'''
Get most likely value of dBx
'''
dBx = np.linspace(-bnd, bnd, N)
realEven = data_Even(np.unwrap(dtr(phaseArray[:,1])))
dBx, dBxMinSqrs = findMin_dBx_OnSpace(dBx, dBy, dBz, realEven, dec, ha)

'''
Get most likely value of dBz
'''
realOdd = data_Odd(np.unwrap(dtr(phaseArray[:,1])))
dBy = np.linspace(-bnd, bnd, N)
dBy, dByMinSqrs = findMin_dBy_OnSpace(dBx, dBy, dBz, realOdd, dec, ha)

'''
Get least squares values for a range of dBz given the dBx and dBy found above
'''
dBz = np.linspace(-bnd, bnd, N)
sqrs12 = []
sqrs33 = []
for i in range(dBz.shape[0]):
    theory12 = getPhase_Theory(dBx, dBy, dBz[i], dtr(12.5), ha)
    theoryEven12 = data_Even(theory12)
    realEven12 = data_Even(np.unwrap(dtr(phaseArray[:,1])))
    sqrs12.append(sumOfSquares(realEven12%(2*pi), theoryEven12%(2*pi)))
    
    theory33 = getPhase_Theory(dBx, dBy, dBz[i], dtr(33.2), ha)
    theoryEven33 = data_Even(theory33)
    realEven33 = data_Even(np.unwrap(dtr(phaseArray[:,2])))
    sqrs33.append(sumOfSquares(realEven33%(2*pi), theoryEven33%(2*pi)))


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
#theory = getPhase_Theory(dBx, dBy, -1.565, dtr(12.5), ha)
#real = dtr(phaseArray[:,1]) - theory
#plt.scatter(phaseArray[:, 0], rtd(real%(2*pi)), color='blue', label='12.5 Dec')
#
#theory = getPhase_Theory(dBx, dBy, -1.565, dtr(33.2), ha)
#real = dtr(phaseArray[:,2]) - theory
#plt.scatter(phaseArray[:, 0], rtd(real%(2*pi)), color='green', label='33.2 Dec')
#plt.legend(prop={'size': 24}, loc=[.2, 1.01], ncol=2)
#plt.xlabel('Hour Angle (in degrees)', fontsize=28, labelpad=20)
#plt.ylabel('Phase (in degrees)', fontsize=28, labelpad=20)

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
>>>>>>> 0025db2d98146882b05c7e0b3165705ee7382836




<<<<<<< HEAD




=======
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
>>>>>>> 0025db2d98146882b05c7e0b3165705ee7382836





<<<<<<< HEAD
=======
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

>>>>>>> 0025db2d98146882b05c7e0b3165705ee7382836





<<<<<<< HEAD



=======
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
      
>>>>>>> 0025db2d98146882b05c7e0b3165705ee7382836
