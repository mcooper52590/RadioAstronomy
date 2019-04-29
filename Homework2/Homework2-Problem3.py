#!/usrbin/env python3e-
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:03:39 2019

@author: mattcooper
"""
import math
import numpy as np

m_e = 9.109e-31
k = 1.38e-23 
c = 3e8
T = 10e7

Bo = np.sqrt(k*T/(m_e*c**2))
s = 3
a = (Bo**2)*(s**2)

#Calculates the ratio of emissivity for different harmonics of the fundamental gyroresonance frequency, 
#not including the sin theta dependance.  S1 and S2 are integer values
def get_emissivity_ratio(s1, s2, T):
    m_e = 9.109e-31
    k = 1.38e-23 
    c = 3e8
    Bo = k*T/(m_e*c**2)
    numer1 = (s2**2)/math.factorial(s2)
    numer2 = ((s2**2)*(Bo**2)/2)
    denom1 = (s1**2)/math.factorial(s1)
    denom2 = ((s1**2)*(Bo**2)/2)
    return  numer1*(numer2**(s2-1))/denom1*(denom2**(s1-1))

emRat = get_emissivity_ratio(1, 4, 10e7)