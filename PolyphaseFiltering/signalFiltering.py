#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:52:54 2019

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

fig, axes = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(17,12), dpi=166) 

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
Channels = 10
fbase = 5
FDMSig = zeros([t.shape[0]+pad, Channels + 1])
FDMSigNoise = zeros([t.shape[0]+pad, Channels + 1])
for i in range(1, 1 + Channels):
  FDMSig[hpad:(t.shape[0]+hpad),i]=(FDMSig[hpad:(t.shape[0]+hpad),i] + 
                                    sin(2*pi*(fbase*+50*i)*t))
  FDMSig[hpad:(t.shape[0]+hpad),0]=(FDMSig[hpad:(t.shape[0]+hpad),0] + 
                                    FDMSig[hpad:(t.shape[0]+hpad),i])
  FDMSigNoise[hpad:(t.shape[0]+hpad),i]=(FDMSigNoise[hpad:(t.shape[0]+hpad),i] + 
                                    (1+srand())*sin(2*pi*(fbase*+50*i)*t + 2*pi*rand()))
  FDMSigNoise[hpad:(t.shape[0]+hpad),0]=(FDMSigNoise[hpad:(t.shape[0]+hpad),0] + 
                                    FDMSigNoise[hpad:(t.shape[0]+hpad),i])


'''
'''
fcarry = 1e4
carrySig = np.zeros([t.shape[0] + pad])
carrySig[hpad:(t.shape[0]+hpad)] = cos(2*pi*fcarry*t)
sigDown = carrySig*FDMSig[:,0]

axes[0,0].plot(tpad, FDMSig[:,0], linewidth=.3, color='black') 
axes[1,0].plot(tpad, carrySig, linewidth=.3, color='black') 
axes[2,0].plot(tpad, sigDown, linewidth=.3, color='black')
axes[0,0].set_xlim(0,.01)
axes[1,0].set_xlim(0,.01)
axes[2,0].set_xlim(0,.01)

'''
'''
N = 4096
hN = int(N/2)
freqs = np.fft.fftfreq(N)
SigFFT = fft(FDMSig[:N,0])
axes[0,1].plot(freqs[:hN], np.abs(SigFFT)[:hN], linewidth=.3, color='black')
axes[0,1].set_yticklabels([])
axes[0,1].tick_params(left="off")

carryFFT = fft(carrySig[:N])
axes[1,1].plot(freqs[:hN], np.abs(carryFFT)[:hN], linewidth=.3, color='black')
axes[1,1].set_yticklabels([])
axes[1,1].tick_params(left="off")

downFFT = fft(sigDown[:N])
axes[2,1].plot(freqs[:hN], np.abs(downFFT)[:hN], linewidth=.3, color='black')
axes[2,1].set_yticklabels([])
axes[2,1].tick_params(left="off")
axes[0,0].set_title('Time Domain')
axes[0,1].set_title('Frequency Domain')
fig.subplots_adjust(hspace=.5)
'''
'''
fig, axes = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(17,12), dpi=166)
axes[0].plot(tpad, FDMSig[:,0], linewidth=.3, color='black')
axes[0].set_xlim(0,.01)

wNy = cta(fNy)
lbp = cta(fcarry - 40)/wNy
ubp = cta(fcarry + 40)/wNy
lbs = cta(lbp - 40)/wNy
ubs = cta(ubp + 40)/wNy

N, Wn = sig.buttord([lbp, ubp], [lbs, ubs], .01, 10, False, fs = sampleFreq)
b, a = sig.butter(N, Wn, 'bandpass', False)
w, h = sig.freqs(b, a, 1000)
axes[1].plot(cta(w), 20 * np.log10(abs(h)), linewidth=.3, color='black')
axes[1].set_title('Butterworth filter frequency response')
axes[1].set_xlabel('Frequency [radians / second]')
axes[1].set_ylabel('Amplitude [dB]')
axes[1].grid(which='both', axis='both')

filtSig = sig.lfilter(b, a, FDMSig[:,0])


fig.subplots_adjust(hspace=.75)
#plt.plot(tpad, FDMSig[:,0], color='black', label='Ideal Signal')  
#plt.plot(tpad, FDMSigNoise[:,0], color='black', linestyle='dashed', label='Noisy Signal')  
#plt.xlabel('Time')
#plt.ylabel('Signal Amplitude')
#plt.legend()
#plt.scatter(tpad, FDMSig[:,0], s=1, color='red')
#N, Wn = sig.buttord(cta(50)/fNy, cta(100)/fNy, .1, 10, False, fs = sampleFreq)
#b, a = sig.butter(N, Wn, 'lowpass', False)
#

#plt.plot(tpad, FDMSig[:,0])
#plt.plot(tpad, filtSig)
#plt.xlim(5.5,5.56)















