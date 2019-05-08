#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:20:07 2019

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
pad = 100
hpad = int(pad/2)
samSpc = 1/sampleFreq
spaceLen = 10
t = np.linspace(0,spaceLen,sampleFreq*spaceLen)
tpad = np.linspace(-hpad*samSpc, spaceLen+(hpad*samSpc), sampleFreq*spaceLen+pad)

'''
'''
Channels = 20
fbase = 5e3
FDMSig = zeros([t.shape[0]+pad, Channels + 1])
FDMSigNoise = zeros([t.shape[0]+pad, Channels + 1])
for i in range(1, 1 + Channels):
  FDMSig[hpad:(t.shape[0]+hpad),i]=(FDMSig[hpad:(t.shape[0]+hpad),i] + 
                                    sin(2*pi*(fbase+10*i)*t))
  FDMSig[hpad:(t.shape[0]+hpad),0]=(FDMSig[hpad:(t.shape[0]+hpad),0] + 
                                    FDMSig[hpad:(t.shape[0]+hpad),i])
  FDMSigNoise[hpad:(t.shape[0]+hpad),i]=(FDMSigNoise[hpad:(t.shape[0]+hpad),i] + 
                                    (1+srand())*sin(2*pi*(fbase*+100*i)*t + 2*pi*rand()))
  FDMSigNoise[hpad:(t.shape[0]+hpad),0]=(FDMSigNoise[hpad:(t.shape[0]+hpad),0] + 
                                    FDMSigNoise[hpad:(t.shape[0]+hpad),i])

fig, axes = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(17,12), dpi=166) 
N = int(1e4)
freqs = np.fft.fftfreq(N, d=samSpc)
wind = sig.windows.blackmanharris(N)
fftSig = fft(FDMSig[:N,0])
fftSigWin = fft(wind*FDMSig[:N,0])

axes[0,0].set_title('Time Domain')
axes[0,0].set_title('Frequency Domain')
axes[0,0].plot(tpad[:N], FDMSig[:N,0], color='black')
axes[0,0].set_xticklabels([])
axes[0,0].tick_params(left="off")

axes[1,0].plot(tpad[:N], wind*FDMSig[:N,0], color='black')

axes[0,1].plot(freqs, np.abs(fftSig), color='black')
axes[0,1].set_xlim(4950,5250)
axes[0,1].set_yticklabels([])
axes[0,1].tick_params(left="off")

axes[1,1].plot(freqs, np.abs(fftSigWin), color='black')
axes[1,1].set_xlim(4950,5250)
axes[1,1].set_yticklabels([])
axes[1,1].tick_params(left="off")

#N = 64*4096
#win = sig.windows.blackmanharris(N)
#fig, axes = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(17,12), dpi=166) 
#axes[0,0].plot(tpad, FDMSig[:,0], linewidth=.3, color='black') 
#axes[1,0].plot(tpad, FDMSig[:,1], linewidth=.3, color='black') 
#axes[0,0].set_xlim(0,.02)
#axes[1,0].set_xlim(0,.005)
#axes[2,0].set_xlim(0,.02)
#
#hN = int(N/2)
#freqs = np.fft.fftfreq(N, samSpc)
#SigFFT = fft(FDMSig[:N,0])
#axes[0,1].plot(freqs[:hN], np.abs(SigFFT)[:hN], linewidth=.1, color='black')
#axes[0,1].set_yticklabels([])
#axes[0,1].tick_params(left="off")
#
#carryFFT = fft(FDMSig[:N,1])
#axes[1,1].plot(freqs[:hN], np.abs(carryFFT)[:hN], linewidth=.1, color='black')
#axes[1,1].set_yticklabels([])
#axes[1,1].tick_params(left="off")
#
#downFFT = fft(win*FDMSig[:N,0])
#axes[2,1].plot(freqs[:hN], np.abs(downFFT)[:hN], linewidth=.1, color='black')
#axes[2,1].set_yticklabels([])
#axes[2,1].tick_params(left="off")
#axes[0,0].set_title('Time Domain')
#axes[0,1].set_title('Frequency Domain')
#fig.subplots_adjust(hspace=.5)