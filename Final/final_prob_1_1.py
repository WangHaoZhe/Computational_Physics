#!/usr/local/bin/python3.11
# -*- coding: UTF-8 -*-
# @Project : Computational_Physics
# @File    : final_prob_1_1.py
# @Author  : Albert Wang
# @Time    : 2024/6/7
# @Brief   : None

import numpy as np
import matplotlib.pyplot as plt

# Constants
S = 13
r_1 = 1.0
r_2 = 0.5  # 0.5 for Problem 1.1, 1.0 for Problem 1.2, 2.0 for Problem 1.3
r_3 = 1.0
a = 0.0
b = 5000.0
h = 0.1

# Generate matrix A
first_diag = np.zeros(S - 1)
second_diag = np.zeros(S - 2)
for i in range(S - 2):
    if i % 2 == 1:
        first_diag[i] = r_2
        second_diag[i] = 0
    else:
        first_diag[i] = r_3
        second_diag[i] = -r_1
first_diag[-1] = r_2
A = (
    np.diag(first_diag, k=1)
    + np.diag(second_diag, k=2)
    + np.diag(-first_diag, k=-1)
    + np.diag(-second_diag, k=-2)
)

# Initial conditions
x_alpha = np.zeros(S)
for alpha in range(S):
    if alpha % 2 == 1:
        x_alpha[alpha] = (2 - 0.01 * (S + 1)) / (S - 1)
    else:
        x_alpha[alpha] = 0.01


# ALVE
def f(x_, A_):
    f_ = np.zeros(S)
    for i in range(S):
        f_[i] = x_[i] * np.dot(A_[i], x_)
    return f_


# Runga-Kutta method
def rk4(x_, A_, h):
    k1 = h * f(x_, A_)
    k2 = h * f(x_ + 0.5 * k1, A_)
    k3 = h * f(x_ + 0.5 * k2, A_)
    k4 = h * f(x_ + k3, A_)
    return x_ + (k1 + 2 * k2 + 2 * k3 + k4) / 6


# Time evolution
tpoints = np.arange(a, b, h)
xpoints = np.zeros((S, tpoints.shape[0]))
for t in range(tpoints.shape[0]):
    xpoints[:, t] = x_alpha
    x_alpha = rk4(x_alpha, A, h)

# Plot 2D
plt.figure(dpi=300)
plt.plot(tpoints, xpoints[12], label=r"$x_{13}$", linewidth=0.5)
plt.plot(tpoints, xpoints[5], label=r"$x_{6}$", linewidth=0.5)
plt.plot(tpoints, xpoints[0], label=r"$x_{1}$", linewidth=0.5)
plt.xlabel("Time")
plt.title(f"Mass Evolutions (skewness = {r_2})")
plt.legend()
plt.grid()
plt.savefig(f"./Graph/mass_evolutions_{r_2}.png")
plt.show()
