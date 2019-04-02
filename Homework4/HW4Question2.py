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
ap_array_bl = np.zeros([size, size])
for i in range(ap_array.shape[0]):
    for j in range(ap_array.shape[1]):
        xDist = i - xOff
        yDist = j - yOff
        if np.sqrt(xDist**2 + yDist**2) <= 50:
           ap_array[i,j] = 1
           if np.sqrt(xDist**2 + yDist**2) >= 10:
               ap_array_bl[i,j] = 1
               
ap_array_bl[509:515,:] = 0
ap_array_bl[:,509:515] = 0
#plt.imshow(ap_array, cmap = 'hot')
                 
ap_fft_array = np.abs(np.fft.fft2(ap_array))**2
ap_fft_array = np.roll(ap_fft_array, 512, axis=0)
ap_fft_array = np.roll(ap_fft_array, 512, axis=1)

ap_fft_array_bl = np.abs(np.fft.fft2(ap_array_bl))**2
ap_fft_array_bl = np.roll(ap_fft_array_bl, 512, axis=0)
ap_fft_array_bl = np.roll(ap_fft_array_bl, 512, axis=1)

beam_ratio = np.abs(ap_fft_array_bl)/np.abs(ap_fft_array)
plt.plot(np.arange(0,3437.75, 3437.75/1024), ap_fft_array_bl[512,:])
plt.plot(np.arange(0,3437.75, 3437.75/1024), ap_fft_array[512,:])
#plt.imshow(beam_ratio[462:562, 462:562], vmin = 0, vmax = 1, cmap='hot')