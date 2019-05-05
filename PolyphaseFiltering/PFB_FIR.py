# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import scipy.signal as sig
def pfb_fir_frontend(x, win_coeffs, M, P):
    W = x.shape[0] / M / P
    x_p = x.reshape((int(W*M), P)).T
    h_p = win_coeffs.reshape((M, P)).T
    x_summed = np.zeros((P, int(M * W - M)))
    for t in range(0, int(M*W-M)):
        x_weighted = x_p[:, t:t+M] * h_p
        x_summed[:, t] = x_weighted.sum(axis=1)
    return x_summed.T

def generate_win_coeffs(M, P, window_fn="hamming"):
    win_coeffs = sig.get_window(window_fn, M*P)
    sinc = sig.firwin(M * P, cutoff=1.0/P, window="rectangular")
    win_coeffs *= sinc
    return win_coeffs
'''
------------------------------------------------------------------------------------------
'''
import numpy as np
import matplotlib.pyplot as plt

sin = np.sin
cos = np.cos
pi=np.pi
rand = np.random.rand
sinc = np.sinc
corr = np.correlate

a = 10
f = 10
t = np.linspace(-10,10,1000)
signal1 = np.zeros([t.shape[0]])
signal2 = np.zeros([t.shape[0]])  
for i in range(500, 600):
    signal1[i] = signal1[i] + .1*sin(.1*i) + (2*rand() - 2*rand())/2
    signal2[i-200] = signal2[i] + .1*sin(.1*i) + (2*rand() - 2*rand())/2

correl = corr(signal1, signal2, 'full')
coeffs = generate_win_coeffs(32,8)
t2 = np.linspace(-10,10,1999)

M = 8
P = 32

#x = np.sin(np.arange(0, M*P*10) / np.pi)
win_coeffs = generate_win_coeffs(M, P, window_fn="hamming")

#plt.subplot(2,1,1)
#plt.title("Time samples")
#plt.plot(x)
#plt.xlim(0, M*P*3)
#
#plt.subplot(2,1,2)
#plt.title("Window function")
#plt.plot(win_coeffs)
#plt.xlim(0, M*P)

#y_p = pfb_fir_frontend(x, win_coeffs, M, P)
#
#plt.figure()
#plt.imshow(y_p)
#plt.xlabel("Branch")
#plt.ylabel("Time")
#
#plt.figure()
#plt.plot(y_p[0], label="p=0")
#plt.plot(y_p[1], label="p=1")
#plt.plot(y_p[2], label="p=2")
#plt.xlabel("Time sample, $n'$")
#plt.legend()
#P = 5
#window = sinc(x)
#winSig = signal*window
#winSigSplt = signal.reshape((int(signal.shape[0]/P), P)).T
#spltSum = winSigSplt.sum(0)
#plt.plot(x, signal)
#
#pltr = np.linspace(-10,10,signal.shape[0]/P)
#M = 4
#P = int(signal.shape[0]/M)
#W = int(signal.shape[0] / M / P)

#

#plt.plot(pltr, spltSum)
#h_p = win_coeffs.reshape((M, P)).T
#x_summed = np.zeros((P, M * W - M))
#for t in range(0, M*W-M):
#    x_weighted = x_p[:, t:t+M] * h_p
#    x_summed[:, t] = x_weighted.sum(axis=1)



