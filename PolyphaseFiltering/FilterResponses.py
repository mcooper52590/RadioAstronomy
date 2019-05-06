#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:54:41 2019

@author: matthew
"""

import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
def cta(freq):
    return 2*pi*freq

fig, axes = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(17,12), dpi=166) 

sampleFreq = int(5e4)
fNy=sampleFreq/2

fcarry = 1e4
wNy = cta(fNy)
lbp = cta(fcarry - 40)/wNy
ubp = cta(fcarry + 40)/wNy
lbs = cta(lbp - 40)/wNy
ubs = cta(ubp + 40)/wNy

N, Wn = sig.buttord([lbp, ubp], [lbs, ubs], .01, 10, False, fs = sampleFreq)
b, a = sig.butter(N, Wn, 'bandpass', False)
w, h = sig.freqs(b, a, 1000)
axes[0,0].plot(w, 20 * np.log10(abs(h)))
axes[0,0].set_title('Butterworth filter frequency response')
axes[0,0].set_xlabel('Frequency [radians / second]')
axes[0,0].set_ylabel('Amplitude [dB]')
#axes[0,0].margins(0, 0.1)
axes[0,0].grid(which='both', axis='both')
#axes[0,0].axvline(100, color='green') # cutoff frequency
