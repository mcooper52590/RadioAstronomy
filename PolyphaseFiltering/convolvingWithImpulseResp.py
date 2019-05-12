#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 11:36:28 2019

@author: matthew
"""

from pylab import *
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
                                    sin(2*pi*(fbase+100*i)*t))
  FDMSig[hpad:(t.shape[0]+hpad),0]=(FDMSig[hpad:(t.shape[0]+hpad),0] + 
                                    FDMSig[hpad:(t.shape[0]+hpad),i])
  FDMSigNoise[hpad:(t.shape[0]+hpad),i]=(FDMSigNoise[hpad:(t.shape[0]+hpad),i] + 
                                    (1+srand())*sin(2*pi*(fbase*+100*i)*t + 2*pi*rand()))
  FDMSigNoise[hpad:(t.shape[0]+hpad),0]=(FDMSigNoise[hpad:(t.shape[0]+hpad),0] + 
                                    FDMSigNoise[hpad:(t.shape[0]+hpad),i])
    
#fig, axes = plt.subplots(2, 1, sharex=False, sharey=False, figsize=(17,12), dpi=166) 

sampleFreq = int(5e4)
fNy=sampleFreq/2

wNy = cta(fNy)
ubp = cta(10000)/wNy
ubs = cta(11000)/wNy

N, Wn = sig.buttord(ubp, ubs, .1, 10, False)
b, a = sig.butter(N, Wn, 'lowpass', False)
l = len(b)
impulse = repeat(0.,l); impulse[0] =1.
x = arange(0,l)
response = sig.lfilter(b,a,impulse)
subplot(211)