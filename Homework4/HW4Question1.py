#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 14:27:13 2019

@author: matthew
"""

import numpy as np
import matplotlib.pyplot as plt

size = 1024
xOff = 512
yOff = 512
ap_array = np.zeros([size, size])
for i in range(ap_array.shape[0]):
    for j in range(ap_array.shape[1]):
        xDist = i - xOff
        yDist = j - yOff
        if np.sqrt(xDist**2 + yDist**2) <= 60:
            ap_array[i,j] = 1
 


                    
ap_fft_array = np.fft.fft2(ap_array)
ap_fft_array = np.roll(ap_fft_array, 512, axis=0)
ap_fft_array = np.roll(ap_fft_array, 512, axis=1)
#plt.imshow(np.abs(ap_fft_array)**2, vmin = 0, vmax = np.abs(ap_fft_array.max()), cmap='hot')
plt.plot(np.arange(0,3437.75, 3437.75/1024), np.abs(ap_fft_array[512,:])**2)
plt.title('1-D Power Profile')
plt.xlabel('Arcminutes')
plt.ylabel('Power')