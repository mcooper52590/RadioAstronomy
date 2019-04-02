#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:32:26 2019

@author: mattcooper
"""

import numpy as np
import matplotlib.pyplot as plt

m_e = 9.109e-31
k = 1.38e-23 
c = 3e8
T = 10e8

Bo = np.sqrt(k*T/(m_e*c**2))

mu = np.arange(10e5,10e8,1e6)
theta = np.arange(0, np.pi, np.pi/8)


s = 2
B = 10
mu_B = 2.8e6*s*B
A = np.zeros([mu.shape[0], theta.shape[0]])
for i in range(0, mu.shape[0]):
    for j in range(0, theta.shape[0]):
        A[i,j] = np.exp(-(((1-(s*mu_B)/mu[i]))**2)/(2*(Bo**2)*np.cos(theta[j])**2))
        
for i in range(0, theta.shape[0]):
    plt.plot(mu, A[:,i])

plt.title('Temperature = 10^8 Kelvin')
plt.show()