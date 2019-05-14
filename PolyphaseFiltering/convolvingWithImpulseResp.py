#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 11:36:28 2019

@author: matthew
"""

import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt

sin = np.sin
cos = np.cos
pi = np.pi
zeros = np.zeros
rand=np.random.rand
fft = np.fft.fft

def srand():
  return rand()-rand()

def cta(freq):
    return 2*pi*freq

'''
'''
sampleFreq = int(5e4)
fNy=sampleFreq/2
pad = 10
hpad = int(pad/2)
samSpc = 1/sampleFreq
spaceLen = 10
t = np.linspace(0,spaceLen,sampleFreq*spaceLen)
tpad = np.linspace(-hpad*samSpc, spaceLen+(hpad*samSpc), sampleFreq*spaceLen+pad)

'''
Create signal with 30 sine functions spaced by 200 Hz.
'''
comps = 60
fbase = 8e3
sep = 100
FDMSig = zeros([t.shape[0]+pad, comps + 1])
FDMSigNoise = zeros([t.shape[0]+pad, comps + 1])
for i in range(1, 1 + comps):
  FDMSig[hpad:(t.shape[0]+hpad),i]=(FDMSig[hpad:(t.shape[0]+hpad),i] + 
                                    sin(2*pi*(fbase+sep*i)*t))
  FDMSig[hpad:(t.shape[0]+hpad),0]=(FDMSig[hpad:(t.shape[0]+hpad),0] + 
                                    FDMSig[hpad:(t.shape[0]+hpad),i])
#  FDMSigNoise[hpad:(t.shape[0]+hpad),i]=(FDMSigNoise[hpad:(t.shape[0]+hpad),i] + 
#                                    (1+srand())*sin(2*pi*(fbase*+sep*i)*t))
#  FDMSigNoise[hpad:(t.shape[0]+hpad),0]=(FDMSigNoise[hpad:(t.shape[0]+hpad),0] + 
#                                    FDMSigNoise[hpad:(t.shape[0]+hpad),i])
    
#fig, axes = plt.subplots(2, 1, sharex=False, sharey=False, figsize=(17,12), dpi=166) 

'''
Create ideal lowpass filter of Butterworth type with order 48
'''
wNy = cta(fNy)
ubp = cta(10000)/wNy
ubs = cta(11275)/wNy

N, Wn = sig.buttord(ubp, ubs, .1, 50, False)
b, a = sig.butter(N, Wn, 'lowpass', False)
l = len(b)
impulse = np.repeat(0.,l); impulse[0] =1.
x = np.arange(0,l)
resp = sig.lfilter(b,a,impulse)


'''
Implementation of a polyphase downsampler
'''
M = 2
N = 4096
signal = FDMSig[:N,0] 
sbl = int(signal.shape[0]/M)
sigBands = np.zeros([sbl,M])
sigBands[:,0] = signal[::M]
for i in range(1,M):
    sigBands[:,M-i] = signal[i::M]

fbl = int(resp.shape[0]/M)
filtBands = np.zeros([fbl, M])
for i in range(0,M):
    filtBands[:,i] = resp[i::M]
    
subConv = np.zeros([sbl + fbl + (M-1), M])
for i in range(0, M):
    subConv[i:subConv.shape[0]-(M-i), i] = sig.convolve(sigBands[:,i], 
            filtBands[:,i], 'full', 'direct')
subConv = subConv.transpose()
polyPhaseOut = subConv.sum(axis=0)
#plt.plot(np.arange(0, 141,1), polyPhaseOut)

'''
Implementation of filtering, then downsampling
'''
filtSig = sig.lfilter(b, a, FDMSig[:,0])
decFiltSig = filtSig[::M]

#=========================================================================================
fig, axes = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(17,12), dpi=166)

'''
Plot FFT with filter implementation and decimation
'''
axes[0,0].plot(tpad[::M], decFiltSig, color='black', linewidth=.5)
#axes[0,0].set_xlim(2.2,2.21)
#axes[0,0].set_ylim(-2.2,2.2)
axes[0,0].set_xticklabels([])
axes[0,0].tick_params(bottom="off")

fftSig = fft(decFiltSig[500:500+N])
freqs = np.fft.fftfreq(decFiltSig[500:500+N].shape[0], d=samSpc*M)
axes[0,1].plot(freqs, np.abs(fftSig), color='black', linewidth=.2)
#axes[0,1].set_yticklabels([])
axes[0,1].tick_params(bottom="off")


'''
Plot polyphase output
'''
axes[1,0].plot(np.linspace(2.2,2.21, len(polyPhaseOut)), polyPhaseOut, 
    color='black', linewidth=.2)
axes[1,0].set_xticklabels([])
axes[1,0].tick_params(bottom="off")

fftPolySig = fft(polyPhaseOut)
freqsPoly = np.fft.fftfreq(polyPhaseOut.shape[0], d=samSpc*M)
axes[1,1].plot(freqsPoly, np.abs(fftPolySig), color='black', linewidth=.5)
#axes[1,1].set_yticklabels([])
axes[1,1].tick_params(left="off")









