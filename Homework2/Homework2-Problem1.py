#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 16:05:26 2019

@author: matthew
"""

u = 1.661e-27

k = 1.38e-23 
c = 3e8
T = 10e8
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

nu_P = get_Photon_Frequency(m_P, 2, 1)
nu_He = get_Photon_Frequency(m_He, 2, 1)
nu_C = get_Photon_Frequency(m_C, 2, 1)

dV_He = (nu_He - nu_P)*c/nu_P
dV_C = (nu_C - nu_P)*c/nu_P