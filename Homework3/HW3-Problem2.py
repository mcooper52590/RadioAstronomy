#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 16:01:22 2019

@author: mattcooper
"""
 
import numpy as np

u = 1.661e-27
k = 1.38e-23 
c = 3e8
m_He = 4.003*u
m_C = 12.0107*u
m_P = 1.673e-27
T = 10e6
nu = 8e8
R_Sun = 6.96e10
h = 6.626e-34

b = T**(3/2)
numer = (T**(3/2))*(nu**2)
denom = .04075*.1*R_Sun
S = 24.5 + np.log(T) - np.log(nu)
a = numer/denom
n_e = np.sqrt(numer/(denom*S)) 


c = 10e4*k/h