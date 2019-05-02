#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 08:52:14 2019

@author: matthew
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig





fft = np.fft.fft
sin=np.sin
cos=np.cos
pi=np.pi
exp = np.exp
log = np.log10

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

'''
Get brick-wall filter coefficients
'''
coeffNum = 300
tWin = np.linspace(-1,1,coeffNum)
filt = np.sinc(tWin)

'''
Implementation of a polyphase downsampler with windowing
'''
M = 4
sbl = int(signal.shape[0]/M)
sigBands = np.zeros([sbl,M])
sigBands[:,0] = signal[::M]
for i in range(1,M):
    sigBands[:,M-i] = signal[i::M]

fbl = int(filt.shape[0]/M)
filtBands = np.zeros([fbl, M])
for i in range(0,M):
    filtBands[:,i] = filt[i::M]
    
subConv = np.zeros([sbl + fbl + (M-1), M])
for i in range(0, M):
    subConv[i:subConv.shape[0]-(M-i), i] = sig.convolve(sigBands[:,i], filtBands[:,i], 'full', 'direct')

#plt.plot(np.arange(0,2525,1), sigBands[:,0], color='red')
#plt.plot(np.arange(0,2525,1), sigBands[:,1], color='green')
#plt.plot(np.arange(0,2525,1), sigBands[:,2], color='blue')
#plt.plot(np.arange(0,2525,1), sigBands[:,3], color='black')
#    
#plt.plot(np.arange(0,2533,1), subConv[:,0], color='red')   
#plt.plot(np.arange(0,2533,1), subConv[:,1], color='green')  
#plt.plot(np.arange(0,2533,1), subConv[:,2], color='blue')  
#plt.plot(np.arange(0,2533,1), subConv[:,3], color='black')  

PF_Dec = np.zeros([subConv.shape[0]])    
for i in range(0, subConv.shape[0]):
    PF_Dec[i] = sum(subConv[i,:])

   
#plt.plot(np.arange(0,2533,1), PF_Dec)

freqs = np.fft.fftfreq(signal[::4].shape[0], 4*samSpc)
mixedSig = sig.windows.blackmanharris(signal.shape[0])*signal
fft1 = fft(mixedSig[::4])

sbfreqs = np.fft.fftfreq(PF_Dec.shape[0], M*samSpc)
fftPF = fft(PF_Dec)


plt.plot(freqs, 20*log(np.abs(fft1)), color='blue')
plt.plot(sbfreqs, 20*log(np.abs(fftPF)), color='red')



'''
Downsampling then windowing for the FFT
'''
decSig = signal[::4]
decWin = sig.windows.blackmanharris(decSig.shape[0])
mixedDec = decSig*decWin

decfreqs = np.fft.fftfreq(decSig.shape[0], 4*samSpc)
fftDec = fft(mixedDec)

plt.plot(decfreqs, 20*log(np.abs(fftDec)), color='green')







#tWin = np.linspace(-2,2,1014 + pad)
#win1 = np.sinc(signal.shape[0])
#mixedSig = np.zeros([signal.shape[0]])
#mixedSig = np.convolve(win1, signal)

#freqs = np.fft.fftfreq(signal.shape[0], spaceLen/signal.shape[0])
#sigfft = fft(signal)
#mixedsigfft = fft(mixedSig)

#plt.plot(freqs, 20*log(np.abs(sigfft)))
#plt.plot(freqs, 20*log(np.abs(mixedsigfft)))
#win2 = sig.windows.blackmanharris(1024)
#mixedSig2 = np.zeros([t.shape[0] + 512])
#mixedSig2[256:(t.shape[0]+256)] = win2*sig1[256:(t.shape[0]+256)]
#plt.plot(t, sig, color='green')
#plt.plot(t, mixedSig1, color='blue')

#freqs = np.fft.fftfreq(sig1.shape[0], .0001)
#fft1 = fft(sig1)
#fftMixed2 = fft(mixedSig2)
#plt.plot(2*freqs, log(np.abs(fft1)), color='green')
#plt.plot(2*freqs, log(np.abs(fftMixed2)), color='blue')







#buttord = sig.buttord(4, 100, .1, 10)
#butter = sig.butter(10, .5, 'low')
#w,h = sig.freqz(butter[0], butter[1])
#plt.plot(w, h)