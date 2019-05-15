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
comps = 10
fbase = 5e3
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
ubp = 5000/fNy
ubs = 7000/fNy

N, Wn = sig.buttord(ubp, ubs, .1, 37, False, fs=sampleFreq)
b, a = sig.butter(N, Wn, 'lowpass', False)
l = 10*len(b)
impulse = np.repeat(0.,l); impulse[0] =1.
x = np.arange(0,l)
resp = sig.lfilter(b,a,impulse)

'''
Implementation of a polyphase downsampler
'''
M = 10
N = 10*4096
signal = FDMSig[500:N+500,0] 
sbl = int(signal.shape[0]/M)
sigBands = np.zeros([sbl + 1,M])
sigBands[:sbl,0] = signal[::M]
for i in range(1,M):
    sigBands[1:,M-i] = signal[i::M]

fbl = int(resp.shape[0]/M)
filtBands = np.zeros([fbl, M])
for i in range(0,M):
    filtBands[:,i] = resp[i::M]
    
subConv = np.zeros([sbl+1, M])
for i in range(0, M):
    subConv[:, i] = sig.lfilter(filtBands[:,i], 1, sigBands[:,i])
#subConv = subConv.transpose()
polyphaseOut = np.sum(subConv, 1)
#plt.plot(np.arange(0, 141,1), polyPhaseOut)

'''
Implementation of filtering, then downsampling
'''
filtSig = sig.lfilter(resp, 1, FDMSig[500:N+500,0])
decFiltSig = filtSig[::M]

#plt.plot(np.arange(0,20,1), filtBands)
#=========================================================================================
fig, axes = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(17,12), dpi=166)

'''
Plot FFT with filter implementation and decimation
'''
axes[0,0].plot(np.arange(0, 4096,1), decFiltSig, color='black', linewidth=.5)
#axes[0,0].set_xlim(2.2,2.21)
#axes[0,0].set_ylim(-2.2,2.2)
axes[0,0].set_xticklabels([])
axes[0,0].tick_params(bottom="off")

fftSig = fft(decFiltSig)
freqs = np.fft.fftfreq(decFiltSig.shape[0], d=samSpc*M)
hl = int(len(freqs)/2)
axes[0,1].plot(freqs[:hl], np.abs(fftSig)[:hl], color='black', linewidth=.2)
#axes[0,1].set_ylim(0,5000)
#axes[0,1].set_yticklabels([])
axes[0,1].tick_params(bottom="off")

'''
Plot polyphase output
'''
axes[1,0].plot(np.linspace(2.2,2.21, len(polyphaseOut)), polyphaseOut, 
    color='black', linewidth=.2)
axes[1,0].set_xticklabels([])
axes[1,0].tick_params(bottom="off")

fftPolySig = fft(polyphaseOut)
freqsPoly = np.fft.fftfreq(polyphaseOut.shape[0], d=samSpc*M)
hlpoly = int(len(freqsPoly)/2)
axes[1,1].plot(freqsPoly[:hlpoly], np.abs(fftPolySig)[:hlpoly], color='black', linewidth=.5)
#axes[1,1].set_yticklabels([])
axes[1,1].tick_params(left="off")


#fftSubs = zeros([freqsSub.shape[0], M])
#for i in range(0, M):
#    fftSubs[:,i] = fft(sigBands[:,i])
#    axes[i].plot(freqsSub, np.real(fftSubs[:,i]))
#axes[M].plot(freqsSub, np.sum(fftSubs, 1))
    
#sigFreqs1 = np.fft.fftfreq(sigBands.shape[0], d=samSpc*M)
#sigFreqs2 = np.fft.fftfreq(subConv.shape[0], d=samSpc*M)
#sigFFT1 = fft(sigBands[:,2])
#sigFFT2 = fft(subConv[:,2])
#plt.plot(sigFreqs1, np.real(sigFFT1))
#plt.plot(sigFreqs2, np.real(sigFFT2))






