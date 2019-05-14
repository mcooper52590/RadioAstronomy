#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:24:16 2019

@author: matthew
"""

import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

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

sampFreq = 1e4
T = 1/sampFreq
M = 30
N = 2*M+1
dw = (2*pi)/(N*T)
n = np.arange(0,1000,.1)
d = sin(pi*n)/sin(pi*n/N)
plt.plot(n,d)

b,a = sig.bessel(6, .1, btype='low')
l = 100
imp = repeat(0.,l); imp[0] =1.
x = arange(0,l)
resp = sig.lfilter(b,a,imp)
#plt.scatter(x, resp)
#plt.plot(x, resp*d)


