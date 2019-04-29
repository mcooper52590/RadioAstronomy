#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 16:01:22 2019

@author: mattcooper
"""
 
import numpy as np
import matplotlib.pyplot as plt
u = 1.661e-27
k = 1.38e-23 
c = 3e8
m_He = 4.003*u
m_C = 12.0107*u
m_P = 1.673e-27
T = 10e6
nu = 10e9
R_Sun = 6.96e10
h = 6.626e-34


def get_turnover_Freq(n, L, T):
    return n*(T**(-3/4))*np.sqrt(1.24e-2*L)


#Function derived for the turnover frequency of the surface brightness
def nu_Function(NU, T, L, N_E):  
    a = L*.04075*(N_E**2)/(T**(3/2))
    b = 24.5 + np.log(T)
    return (NU**2) + a*np.log(NU) - a*b

#Iterative method to solve for the turnover frequency.  That darn logarithm
#A and B are the lower and upper bounds of the domain, respectively.  Pass any other parameters that may be
#required to compute the function as *args
def bisect_Function(FUNC, A, B, ERR, *args):
    flag = False
    while flag == False:
        M = (A + B)/2
        
        func_A = FUNC(A, *args)
        func_B = FUNC(B, *args)
        func_M = FUNC(M, *args)
        if B - M < ERR:
            print('Error reached')
            return M
        if func_A == func_B:
            print('Bisection method failed.  Function has same sign at both bounds.')
            break
        if func_A == 0 or func_B == 0 or func_M == 0:
            flag = True
        if np.sign(func_A) != np.sign(func_M):
            B = M
        else:
            A = M


n_e1 = 10e13
T1 = 1e4
L1 = 1000000
nu_Turnover_1 = bisect_Function(nu_Function, 1e8, 1e14, 1e7, T1, L1, n_e1)

n_e2 = 10e10
T2 = 2e6
L2 = 10000000
nu_Turnover_2 = bisect_Function(nu_Function, 1e8, 1e14, 1e7, T2, L2, n_e2)

n_e3 = 10e9
T3 = 5e5
L3 = 70000000
nu_Turnover_3 = bisect_Function(nu_Function, 1e8, 1e14, 1e7, T3, L3, n_e3)


freqs = np.arange(10e6, 10e11, 1e7)
tau = np.zeros([freqs.shape[0], 3])
T = np.zeros([freqs.shape[0], 3])
S = 24.5 + np.log(T) - np.log(nu)
for i in range(freqs.shape[0]):
    tau[i, 0] = (n_e1**2)*(1/T1**(3/2))*L1*1.24e-2*(1/(freqs[i]**(2)))*(18.2 + np.log(T1**(3/2)) - np.log(freqs[i]))
    tau[i, 1] = (n_e2**2)*(1/T2**(3/2))*1.24e-2*(1/(freqs[i]**(2)))*L2*(24.5 + np.log(T2) - np.log(freqs[i]))
    tau[i, 2] = (n_e3**2)*(1/T3**(3/2))*1.24e-2*(1/(freqs[i]**(2)))*L3*(24.5 + np.log(T3) - np.log(freqs[i]))

T[:,0] = T1*(1-np.exp(-tau[:,0]))    
T[:,1] = T[:,0]*np.exp(-tau[:,1]) + T2*(1-np.exp(-tau[:,1]))
T[:,2] = T[:,1]*np.exp(-tau[:,2]) + T3*(1-np.exp(-tau[:,2]))

fig1, (ax1) = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(17,12), dpi=166)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.plot(freqs, T[:,0], color='black', label='1st Layer Spectra')
ax1.plot(freqs, T[:,1], color='red', label='2nd Layer Spectra')
ax1.plot(freqs, T[:,2], color='blue', label='3rd Layer Spectra')
ax1.set_ylim(1e3, 1e7)
ax1.set_xlim(1e9, 1e11)
ax1.legend()









