#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:54:41 2019

@author: matthew
"""
from pylab import *
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
def cta(freq):
    return freq/(2*pi)


    
fig, axes = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(17,12), dpi=166) 

sampleFreq = int(5e4)
fNy=sampleFreq/2

wNy = cta(fNy)
cf = 5000
bw = 100
tw = 100
ubp = cta(10000)/wNy
#lbp = cta(cf - bw)/wNy
ubs = cta(12000)/wNy
#lbs = cta(cf - bw - tw)/wNy

N, Wn = sig.buttord(ubp, ubs, .05, 100, False)
b, a = sig.butter(N, Wn, 'lowpass', False)

#l = len(b)
#imp = repeat(0.,l)
#imp[0] =1
#x = arange(0,l)
#resp = sig.lfilter(b,a,imp)
#axes.scatter(x, resp, color='black')
#axes.set_ylabel('Amplitude')
#axes.set_xlabel(r'n (samples)')
#axes.set_title(r'Impulse response')

w, h = sig.freqs(b, a, 200)
plt.xscale('log')
plt.plot(w, 20 * np.log10(abs(h)), color='black')
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.axvline(ubp, color='black', linestyle='dashed')
plt.axvline(ubs, color='black', linestyle='dashed')
plt.grid(which='both', axis='both')


