#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 14:27:13 2019

@author: matthew
"""

import numpy as np
import matplotlib.pyplot as plt

sqrt = np.sqrt

#Transforms a vector in the ENU coordinate system to the XYZ coordinate system
def ENU_To_XYZ(coordVec, S0):
    transMat = np.array([(0,-np.sin(S0), np.cos(S0)),
                        (1 , 0, 0),
                        (0, np.cos(S0), np.sin(S0))])
    return np.matmul(transMat, coordVec)


#Transforms a baseline vector in the XYZ coordinate system to the UVW spatial frequency system
def XYZ_To_UVW(baseVec, h0, S0, Lambda):
    transMat2 = np.array([(np.sin(h0), np.cos(h0), 0),
                        (-np.sin(S0)*np.cos(h0), np.sin(S0)*np.sin(h0), np.cos(S0)),
                        (np.cos(S0)*np.cos(h0), -np.cos(S0)*np.sin(h0), np.sin(S0))]) 
    return (1/Lambda)*np.matmul(transMat2, baseVec)


#Calculates separation vector between two input vectors
def GetSepVec(vec1, vec2):
    return np.array([vec2[0] - vec1[0], vec2[1] - vec1[1], vec2[2] - vec1[2]])

def get_UVW_List(radioArray):
    holder = []
    for key1 in radioArray['AntBLs_UVW'].keys():
        for key2 in radioArray['AntBLs_UVW'][key1].keys():
            if key2 != key1:
                for key3 in radioArray['AntBLs_UVW'][key1][key2].keys():
                    if key2 != key3:
                        holder.append([radioArray['AntBLs_UVW'][key1][key2][key3], 
                                       np.linalg.norm(radioArray['AntBLs_XYZ'][key1][key2])])
    return holder

def get_radioArray_Dict(nu, latitude, S0, hstart, hstop):
    hourArray = np.arange(hstart*15, hstop*15, .1)
    Lambda = 3e8/nu
    #Declare a dict to store information pertaining to the radio array given
    radioArray = {}
    
    #Store antennae locations in a dict within the radioArray dict
    radioArray['AntLoc_ENU'] = {}
    radioArray['AntLoc_ENU'][1] = np.array([187.86, 71.74, 1.04])
    radioArray['AntLoc_ENU'][2] = np.array([196.15, 75.14, 1.03])
    radioArray['AntLoc_ENU'][3] = np.array([175.11, 77.39, 1.04])
    radioArray['AntLoc_ENU'][4] = np.array([197.96, 50.25, 1.04])
    radioArray['AntLoc_ENU'][5] = np.array([194.66, 108.86, 1.13])
    radioArray['AntLoc_ENU'][6] = np.array([147.42, 35.91, 1.04])
    radioArray['AntLoc_ENU'][7] = np.array([266.83, 67.10, 1.23])
    radioArray['AntLoc_ENU'][8] = np.array([98.95, 169.34, 1.32])
    radioArray['AntLoc_ENU'][9] = np.array([20.35, -218.49, .93])
    radioArray['AntLoc_ENU'][10] = np.array([167.43, 280.78, 1.66])
    radioArray['AntLoc_ENU'][11] = np.array([-442.00, -138.59, -0.25])
    radioArray['AntLoc_ENU'][12] = np.array([640.22, -355.82, 0.95])
    radioArray['AntLoc_ENU'][13] = np.array([-329.06, 861.82, 3.01])
    radioArray['AntLoc_ENU'][14] = np.array([-631.00, -184.00, 25.84])
    radioArray['AntLoc_ENU'][15] = np.array([-213.00, -187.00, 25.22])
    
    
    #Get antennae locations in XYZ coordinates
    radioArray['AntLoc_XYZ'] = {}
    for key in radioArray['AntLoc_ENU'].keys():
        radioArray['AntLoc_XYZ'][key] = ENU_To_XYZ(radioArray['AntLoc_ENU'][key], latitude)
    
    
    #Calculate baseline vectors
    radioArray['AntBLs_XYZ'] = {}
    for key1 in radioArray['AntLoc_XYZ']:
        radioArray['AntBLs_XYZ'][key1] = {}
        for key2 in radioArray['AntLoc_XYZ'].keys():
            if key2 != key1:
                radioArray['AntBLs_XYZ'][key1][key2] = GetSepVec(radioArray['AntLoc_XYZ'][key1], 
                                                                           radioArray['AntLoc_XYZ'][key2])
    
    #Transform the baseline vectors to the UVW coordinate system, taking the phase center into account, for a single
    #hour angle 
    radioArray['AntBLs_UVW'] = {}
    for key1 in radioArray['AntBLs_XYZ'].keys():
        radioArray['AntBLs_UVW'][key1] = {}
        for key2 in radioArray['AntBLs_XYZ'][key1].keys():
            if key2 != key1:
                radioArray['AntBLs_UVW'][key1][key2] = {}
                for h0 in hourArray:
                    radioArray['AntBLs_UVW'][key1][key2][round(h0, 1)] = (
                    XYZ_To_UVW(radioArray['AntBLs_XYZ'][key1][key2], h0, S0, Lambda))

    return radioArray


radioArray = get_radioArray_Dict(1e10, 37.23317, 13, -6, 6)

#Pulls the list of (u,v,w) vectors from the dict
holder = get_UVW_List(radioArray)              
                
size = 512
uOff = int(size/2)
vOff = int(size/2)

width = 300000
uShift = width/2
vShift = width/2

du = width/size
dv = width/size


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(12,6))  
props = dict(boxstyle='round', facecolor='white', alpha=0.5)  
ft = 9

#Need to change radius to an ellipse in the even of unequal spacing
#Currently configured for only one neighbor and equal spacing
R = 2*(3/sqrt(2))*585.9375
n = 2

distArrFFT = np.zeros([size, size])
for [(u,v,w),BLWidth] in holder:
    i = int(round(u/du)) + uOff
    j = int(round(v/dv)) + vOff
    for l in range(i - n, i + n):
        for m in range(j - n, j + n):
            rdiff = sqrt((du*l - (u + uShift))**2 + (du*m - (v + vShift))**2)
            distArrFFT[l,m] += (R - rdiff)/R
               
distArr = np.roll(distArrFFT, uOff, axis=0)
distArr = np.roll(distArr, vOff, axis=1)
distArr = np.fft.ifft2(distArr)
distArr = np.roll(distArr, uOff, axis=0)
distArr = np.roll(distArr, vOff, axis=1)

import scipy.io as io
matDict = {}
io.loadmat('/home/matthew/Downloads/uv_model.mat', mdict = matDict)
skyArrFFT = matDict['uv']

#uniArr= np.roll(uniArrFFT, uOff, axis=0)
#uniArr = np.roll(uniArr, vOff, axis=1)
#uniArr = np.fft.ifft2(uniArr)                   
#uniArr= np.roll(uniArr, uOff, axis=0)
#uniArr = np.roll(uniArr, vOff, axis=1)                  



ax1.imshow(np.abs(distArrFFT), cmap='Greys_r')
#Removes all ticks and labels from plot
ax1.tick_params(
    axis='both',          
    which='both',      
    bottom=False,      
    top=False, 
    left=False,
    right=False, 
    labelbottom=False,
    labelleft=False)      
ax1.text(0.05, 0.95, 'I(l,m)', transform=ax1.transAxes, fontsize=ft,
        verticalalignment='top', bbox=props)
ax1.text(0.05, 0.05, 'Map', transform=ax1.transAxes, fontsize=ft,
        verticalalignment='bottom', bbox=props)  

ax2.imshow(np.abs(distArr), cmap='Greys_r')
#Removes all ticks and labels from plot
ax2.tick_params(
    axis='both',          
    which='both',      
    bottom=False,      
    top=False, 
    left=False,
    right=False, 
    labelbottom=False,
    labelleft=False)      
ax2.text(0.05, 0.95, 'I(l,m)', transform=ax2.transAxes, fontsize=ft,
        verticalalignment='top', bbox=props)
ax2.text(0.05, 0.05, 'Map', transform=ax2.transAxes, fontsize=ft,
        verticalalignment='bottom', bbox=props)  

distArrFFT = np.roll(distArrFFT, uOff, axis=0)
distArrFFT = np.roll(distArrFFT, vOff, axis=1)
intensArrUni = np.fft.ifft2(skyArrFFT*distArrFFT)
intensArrUni = np.roll(intensArrUni, uOff, axis=0)
intensArrUni = np.roll(intensArrUni, vOff, axis=1)

ax3.imshow(np.abs(intensArrUni), cmap='Greys_r', vmax = (np.abs(intensArrUni)).max())
#Removes all ticks and labels from plot
ax3.tick_params(
    axis='both',          
    which='both',      
    bottom=False,      
    top=False, 
    left=False,
    right=False, 
    labelbottom=False,
    labelleft=False)      
ax3.text(0.05, 0.95, 'I(l,m)', transform=ax3.transAxes, fontsize=ft,
        verticalalignment='top', bbox=props)
ax3.text(0.05, 0.05, 'Map', transform=ax3.transAxes, fontsize=ft,
        verticalalignment='bottom', bbox=props)  






#natArrFFT = np.zeros([size, size])
#uniArrFFT = np.zeros([size, size])
#for [(u,v,w),BLWidth] in holder:
#    i = int(round(u/du)) + uOff
#    j = int(round(v/dv)) + vOff
#    #Natural Weighting
#    natArrFFT[i,j] += 1
#    #Uniform Weighting
#    uniArrFFT[i,j] = 1
#
#
##Move the u, v plane to where the center is at zero
#natArrFFT = np.roll(natArrFFT, uOff, axis=0)
#natArrFFT = np.roll(natArrFFT, vOff, axis=1)
#natArr = np.fft.ifft2(natArrFFT)
#
#uniArrFFT = np.roll(uniArrFFT, uOff, axis=0)
#uniArrFFT = np.roll(uniArrFFT, vOff, axis=1)
#uniArr = np.fft.ifft2(uniArrFFT)                   
#                    
#                    
##Plots the 
#axes[0,0].imshow(np.log(np.abs(uniArr)), cmap='Greys_r')
##Removes all ticks and labels from plot
#axes[0,0].tick_params(
#    axis='both',          
#    which='both',      
#    bottom=False,      
#    top=False, 
#    left=False,
#    right=False, 
#    labelbottom=False,
#    labelleft=False)      
#axes[0,0].text(0.05, 0.95, 'I(l,m)', transform=axes[0,0].transAxes, fontsize=ft,
#        verticalalignment='top', bbox=props)
#axes[0,0].text(0.05, 0.05, 'Map', transform=axes[0,0].transAxes, fontsize=ft,
#        verticalalignment='bottom', bbox=props)    
#
##Plots the 
#axes[0,1].imshow(np.log(np.abs(natArr)), cmap='Greys_r')
##Removes all ticks and labels from plot
#axes[0,1].tick_params(
#    axis='both',          
#    which='both',      
#    bottom=False,      
#    top=False, 
#    left=False,
#    right=False, 
#    labelbottom=False,
#    labelleft=False)      
#axes[0,1].text(0.05, 0.95, 'I(l,m)', transform=axes[0,1].transAxes, fontsize=ft,
#        verticalalignment='top', bbox=props)
#axes[0,1].text(0.05, 0.05, 'Map', transform=axes[0,1].transAxes, fontsize=ft,
#        verticalalignment='bottom', bbox=props) 
#
##Plots the 
#axes[0,2].imshow(np.log(np.abs(distArr)), cmap='Greys_r')
##Removes all ticks and labels from plot
#axes[0,2].tick_params(
#    axis='both',          
#    which='both',      
#    bottom=False,      
#    top=False, 
#    left=False,
#    right=False, 
#    labelbottom=False,
#    labelleft=False)      
#axes[0,2].text(0.05, 0.95, 'I(l,m)', transform=axes[0,2].transAxes, fontsize=ft,
#        verticalalignment='top', bbox=props)
#axes[0,2].text(0.05, 0.05, 'Map', transform=axes[0,2].transAxes, fontsize=ft,
#        verticalalignment='bottom', bbox=props) 
#
#import scipy.io as io
#import scipy.signal as sig
#matDict = {}
#io.loadmat('/home/matthew/Downloads/uv_model.mat', mdict = matDict)
#skyArrFFT = matDict['uv']
#
#intensArrUni = np.fft.ifft2(skyArrFFT*uniArrFFT)
#intensArrUni = np.roll(intensArrUni, uOff, axis=0)
#intensArrUni = np.roll(intensArrUni, vOff, axis=1)
#
#intensArrNat = np.fft.ifft2(skyArrFFT*natArrFFT)
#intensArrNat = np.roll(intensArrNat, uOff, axis=0)
#intensArrNat = np.roll(intensArrNat, vOff, axis=1)
#
#intensArrDist = np.fft.ifft2(skyArrFFT*distArrFFT)
#intensArrDist = np.roll(intensArrDist, uOff, axis=0)
#intensArrDist = np.roll(intensArrDist, vOff, axis=1)
#
##Plots the 
#axes[1,0].imshow(np.abs(intensArrUni), cmap='Greys_r')
##Removes all ticks and labels from plot
#axes[1,0].tick_params(
#    axis='both',          
#    which='both',      
#    bottom=False,      
#    top=False, 
#    left=False,
#    right=False, 
#    labelbottom=False,
#    labelleft=False)      
#axes[1,0].text(0.05, 0.95, 'I(l,m)', transform=axes[1,0].transAxes, fontsize=ft,
#        verticalalignment='top', bbox=props)
#axes[1,0].text(0.05, 0.05, 'Map', transform=axes[1,0].transAxes, fontsize=ft,
#        verticalalignment='bottom', bbox=props) 
#
##Plots the 
#axes[1,1].imshow(np.abs(intensArrNat), cmap='Greys_r')
##Removes all ticks and labels from plot
#axes[1,1].tick_params(
#    axis='both',          
#    which='both',      
#    bottom=False,      
#    top=False, 
#    left=False,
#    right=False, 
#    labelbottom=False,
#    labelleft=False)      
#axes[1,1].text(0.05, 0.95, 'I(l,m)', transform=axes[1,1].transAxes, fontsize=ft,
#        verticalalignment='top', bbox=props)
#axes[1,1].text(0.05, 0.05, 'Map', transform=axes[1,1].transAxes, fontsize=ft,
#        verticalalignment='bottom', bbox=props) 
#
##Plots the 
#axes[1,2].imshow(np.abs(intensArrDist), cmap='Greys_r', vmax = np.abs(intensArrUni).max())
##Removes all ticks and labels from plot
#axes[1,2].tick_params(
#    axis='both',          
#    which='both',      
#    bottom=False,      
#    top=False, 
#    left=False,
#    right=False, 
#    labelbottom=False,
#    labelleft=False)      
#axes[1,2].text(0.05, 0.95, 'I(l,m)', transform=axes[1,2].transAxes, fontsize=ft,
#        verticalalignment='top', bbox=props)
#axes[1,2].text(0.05, 0.05, 'Map', transform=axes[1,2].transAxes, fontsize=ft,
#        verticalalignment='bottom', bbox=props) 
#fig.subplots_adjust(wspace=0.01, hspace=0.01)













#Panel 1 



#Panel 2
#skyArrFFT = np.roll(skyArrFFT, xOff, axis=0)
#skyArrFFT = np.roll(skyArrFFT, xOff, axis=1)
#axes[1,0].imshow(np.log(np.abs(skyArrFFT)), cmap='Greys_r')
#axes[1,0].tick_params(
#    axis='both',          
#    which='both',      
#    bottom=False,      
#    top=False, 
#    left=False,
#    right=False, 
#    labelbottom=False,
#    labelleft=False)        
#axes[1,0].text(0.05, 0.95, 'V(u,v)', transform=axes[1,0].transAxes, fontsize=ft,
#        verticalalignment='top', bbox=props)
#axes[1,0].text(0.05, 0.05, 'Visbility', transform=axes[1,0].transAxes, fontsize=ft,
#        verticalalignment='bottom', bbox=props)                   
#                    
##Panel 3
#natArr = np.roll(natArr, xOff, axis=0)
#natArr = np.roll(natArr, xOff, axis=1)
#axes[0,1].imshow(np.abs(natArr), cmap='Greys_r')
#axes[0,1].tick_params(
#    axis='both',          
#    which='both',      
#    bottom=False,      
#    top=False, 
#    left=False,
#    right=False, 
#    labelbottom=False,
#    labelleft=False)        
#axes[0,1].text(0.05, 0.95, 'B(l,m)', transform=axes[0,1].transAxes, fontsize=ft,
#        verticalalignment='top', bbox=props)
#axes[0,1].text(0.05, 0.05, 'Beam', transform=axes[0,1].transAxes, fontsize=ft,
#        verticalalignment='bottom', bbox=props)                   
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    