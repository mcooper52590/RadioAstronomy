#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 16:05:26 2019

@author: matthew
"""
import matplotlib.pyplot as plt
import numpy as np

u = 1.661e-27

k = 1.38e-23 
c = 3e8
m_He = 4.003*u
m_C = 12.0107*u
m_P = 1.673e-27


def get_Photon_Frequency(NUC_MASS, N_STRT, N_END):
    m_e = 9.109e-31
    e = 1.602e-19
    h = 6.626e-34
    e_o = 8.85e-12
    c = 3e8
    r_H = ((NUC_MASS*m_e)/(NUC_MASS+m_e))*((e**4)/(8*c*(e_o**2)*(h**3)))
    n_12 = r_H*c*((1/(N_END**2))-(1/(N_STRT**2)))
    return n_12

levels = []
band_92_96 = []
band_104_108 = []
band_227_235 = []
band_243_251 = []
for n in range(1, 1000):
    freqs = [get_Photon_Frequency(m_P, n+1, n), get_Photon_Frequency(m_P, n+2, n), get_Photon_Frequency(m_P, n+3, n)]
    for i, freq in enumerate(freqs):
        if 92e9 < freq < 96e9:
            band_92_96.append([freq, n, i])
        if 103e9 < freq < 108e9:
            band_104_108.append([freq, n, i])
        if 227e9 < freq < 234e9:
            band_227_235.append([freq, n, i])
        if 245e9 < freq < 250e9:
            band_243_251.append([freq, n, i])

T = 10e7
f_o = 232.02       
df = np.sqrt((8*k*T*np.log(2))/(m_P*(c**2)))*f_o