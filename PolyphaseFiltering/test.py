#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:22:51 2019

@author: matthew
"""

import numpy as np

a = np.zeros([10000])
a[2:10:2] = np.arange(1,5,1)