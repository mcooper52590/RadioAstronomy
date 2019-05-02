#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:17:35 2019

@author: matthew
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def srand():
  return rand()-rand()


def add_noise(signal):
  for i in range(signal.shape[0]):
    signal[i] = signal[i] + srand()
  return signal

fft = np.fft.fft
sin=np.sin
cos=np.cos
pi=np.pi
exp = np.exp
log = np.log10
rand=np.random.rand

sampleFreq = 1000
fNy=sampleFreq/2
f=100
w=2*pi*f
pad = 100
hpad = int(pad/2)
samSpc = 1/sampleFreq
spaceLen = 10
t = np.linspace(0,spaceLen,sampleFreq*spaceLen)

'''
Create signal
'''
signal = np.zeros([t.shape[0] + pad])
signal[hpad:(t.shape[0]+hpad)] = sin(w*t + pi/2)
#signal = add_noise(signal)

'''
Get brick-wall filter coefficients
'''
tWin = np.linspace(-2,2,signal.shape[0])
win = np.sinc(tWin)

freqs = np.fft.fftfreq(signal[::4].shape[0], 4*samSpc)
mixedSigBH = sig.windows.blackmanharris(signal.shape[0])*signal
mixedSigSinc = win*signal

M = 4
bl = int(signal.shape[0]/M)
bands = np.zeros([bl, M])
for i in range(0,M):
    bands[:,i] = signal[i*bl:(i+1)*bl]*win[i*bl:(i+1)*bl]
    
decSig = np.zeros([bl])
for i in range(0, decSig.shape[0]):
    decSig[i] = np.sum(bands[i,:])
    
fftBH = fft(mixedSigBH[::4])
fftSinc = fft(decSig)

#plt.plot(freqs, 20*log(np.abs(fftBH)), color='blue')
plt.plot(4*freqs[:int(len(freqs)/2)], 20*log(np.abs(fftSinc))[:int(len(freqs)/2)], color='red')

'''
Downsampling then windowing for the FFT
'''
decSig = signal[::4]
decWin = sig.windows.blackmanharris(decSig.shape[0])
mixedDec = decSig*decWin

decfreqs = np.fft.fftfreq(decSig.shape[0], 4*samSpc)
fftDec = fft(mixedDec)

plt.plot(decfreqs[:int(len(decfreqs)/2)], 20*log(np.abs(fftDec))[:int(len(decfreqs)/2)], color='green')