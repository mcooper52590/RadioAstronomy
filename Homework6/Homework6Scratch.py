#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:08:12 2019

@author: matthew
"""

import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-1e-7,1e-7,1000)
f = 1e10
w = 2*np.pi*f
amp1 = 2
amp2 = 2
#for i in range(0,32):
i = 4
tau = i*np.pi/2
A = amp1*np.sin(w*t)
B = amp2*np.sin(w*t - w*tau)
inten = A*B
plt.plot(t,inten)
