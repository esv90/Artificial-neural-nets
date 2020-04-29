# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:24:04 2020

@author: a339594
"""

import numpy as np

import ann

input1 = ann.Input(5)
dense1 = ann.Dense(5)
model1 = ann.Sequential()

model1.add(input1)