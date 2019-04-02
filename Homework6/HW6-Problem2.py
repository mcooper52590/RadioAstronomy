#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:32:16 2019

@author: matthew
"""

import numpy as np
import matplotlib.pyplot as plt

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
                    XYZ_To_UVW(radioArray['AntBLs_XYZ'][key1][key2], S0, h0, Lambda))

    return radioArray


radioArray = get_radioArray_Dict(1e10, 37.23317, 13, -6, 6)

holder = []
for key1 in radioArray['AntBLs_UVW'].keys():
    for key2 in radioArray['AntBLs_UVW'][key1].keys():
        if key2 != key1:
            for key3 in radioArray['AntBLs_UVW'][key1][key2].keys():
                if key2 != key3:
                    holder.append(radioArray['AntBLs_UVW'][key1][key2][key3])
                    

size = 512
xOff = int(size/2)
yOff = int(size/2)
width = 300000
du = width/size
ap_array_nat = np.zeros([size, size])
ap_array_uni = np.zeros([size, size])
for uvwVec in holder:
    i = int(round(uvwVec[0]/du)) + xOff
    j = int(round(uvwVec[1]/du)) + yOff
    #Natural Weighting
    ap_array_nat[i,j] += 1
    #Uniform Weighting
    ap_array_uni[i,j] = 1

def format_func(value, tick_number):
  return '{0:.3g}'.format(value*1.14e-2)
import matplotlib.ticker as ticker  
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(17,12), dpi=166) 
fig.suptitle(r'Decimated Sampling Function for Uniform and Natural Weighting, 13 Declination', 
             fontsize=12)          


x = np.linspace(-xOff/2, xOff/2, size/2)
y = np.linspace(-yOff/2, yOff/2, size/2)
X, Y = np.meshgrid(x, y)
        

ap_fft_uni = np.fft.fft2(ap_array_uni)
ap_fft_uni = np.roll(ap_fft_uni, xOff, axis=0)
ap_fft_uni = np.roll(ap_fft_uni, yOff, axis=1)
uniAbs = np.abs(ap_fft_uni)

levels = [uniAbs.max()/2, uniAbs.max()/1.00001]
CS = ax1.contour(X, Y, uniAbs[128:384, 128:384], levels, colors = ('green', 'blue'))
extent = (-128, 128, -128, 128)
ax1.imshow(uniAbs[128:384, 128:384], vmin = 0, vmax = uniAbs.max(), cmap='hot', extent = extent)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))


ap_fft_nat = np.fft.fft2(ap_array_nat)
ap_fft_nat = np.roll(ap_fft_nat, xOff, axis=0)
ap_fft_nat = np.roll(ap_fft_nat, yOff, axis=1)
natAbs = np.abs(ap_fft_nat)

levels = [natAbs.max()/2, natAbs.max()/1.00001]
CS = ax2.contour(X, Y, natAbs[128:384, 128:384], levels, colors = ('green', 'blue'))
extent = (-128, 128, -128, 128)
ax2.imshow(natAbs[128:384, 128:384], vmin = 0, vmax = natAbs.max(), cmap='hot', extent = extent)
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))



#CB = fig.colorbar(CS, ax = ax1)
#plt.clabel(CS, inline=1, fontsize=1)
#ax1.colorbar()
#plt.show()
#plt.plot(np.arange(0,3437.75, 3437.75/1024), np.abs(ap_fft_array[512,:])**2)
#plt.title('1-D Power Profile')
#plt.xlabel('Arcminutes')
#plt.ylabel('Power')



















#fig, (ax1) = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(17,12), dpi=166) 
#holder = np.array(holder)
#ax1.scatter(holder[:, 0], holder[:,1], s = .1)
#ax1.set_xlim(-1.5e5, 1.5e5)
#ax1.set_xlabel('U')
#ax1.set_ylim(-1.5e5, 1.5e5)
#ax1.set_ylabel('V')

#                ax1.scatter(radioArray['AntennaeBaselines_UVW'][key1][key2][round(h0, 1)][0], 
#                            radioArray['AntennaeBaselines_UVW'][key1][key2][round(h0, 1)][1])


















