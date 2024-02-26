#!/usr/local/bin/python3.11
# -*- coding: UTF-8 -*-
# @Project : Computational_Physics
# @File    : Problem1_2.py
# @Author  : Albert Wang
# @Time    : 2024/2/26
# @Brief   : None

import numpy as np
import pylab

data = np.loadtxt("stm.txt", float)  # Load file
pylab.imshow(data, origin="lower")  # Plot
pylab.show()

