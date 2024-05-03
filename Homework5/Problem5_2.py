#!/usr/local/bin/python3.11
# -*- coding: UTF-8 -*-
# @Project : Computational_Physics
# @File    : Problem5_2.py
# @Author  : Albert Wang
# @Time    : 2024/5/3
# @Brief   : None

import numpy as np

from matplotlib import pyplot as plt


data = np.loadtxt("dow.txt", float)  # Load file

c = np.fft.rfft(data)  # Discrete Fourier Transform
c[int(len(c) * 0.1) :] = 0
# c[int(len(c) * 0.02) :] = 0
data_inverse = np.fft.irfft(c)

plt.plot(data, label="Original")  # Plot
plt.plot(data_inverse, label="Smoothed")
plt.xlabel("Days")
plt.ylabel("Value")
plt.title("Dow Jones Industrial Average")
plt.legend()
plt.show()
