# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

c = 3e8
k = 1.38e-23 
A = np.pi*(.75**2)
T = 3700
eta = .6

#Total Atmospheric Extinction
Tatmos = 300
RHS = 2*k*Tatmos*((212e9/c)**2)
extinc = RHS/ 1 + RHS


S = (2*k*T)/(eta*A*1e-22)

tau = .58
E = 48*np.pi/180
Tex = 1900
Torig = (Tex - 300*(1-np.exp(-tau*(1/np.sin(E)))))/np.exp(-tau*(1/np.sin(E)))

R = 1.49e11
S_orig = (2*k*((212e9)**2)*Torig)/(c**2* 4*np.pi)