#!/usr/local/bin/python3.11
# -*- coding: UTF-8 -*-
# @Project : Computational_Physics
# @File    : final_prob_1_8.py
# @Author  : Albert Wang
# @Time    : 2024/6/7
# @Brief   : None

import numpy as np
import matplotlib.pyplot as plt

# Constants
S = 69
r_1 = 1.0
r_2 = 1.0  # 1.0 for Problem 1.8, 0.5 for Problem 1.9, 2.0 for Problem 1.10
r_3 = 1.0
a = 0.0
b = 240.0
h = 0.01

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
x_alpha = np.full((S,), 0.01)
x_alpha[5] = x_alpha[12] = 0.01 * 0.45
x_alpha[6] = x_alpha[10] = 0.15 * 0.45
x_alpha[7] = x_alpha[9] = 0.05 * 0.45
x_alpha[8] = 0.3 * 0.45
x_alpha[21] = x_alpha[27] = 0.005 * 0.45
x_alpha[22] = x_alpha[26] = 0.07 * 0.45
x_alpha[23] = x_alpha[25] = 0.015 * 0.45
x_alpha[24] = 0.1 * 0.45


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
alpha = np.arange(1, S + 1, 1)

# Plot 3D
fig = plt.figure(dpi=300)
ax = fig.add_subplot(projection="3d")
for i in [240, 180, 120, 60, 1]:
    ax.plot(alpha, i * np.ones(S), xpoints[:, int(i / h - 1)])
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("Time")
ax.set_zlabel(r"$x_{\alpha}$")
ax.set_yticks([1, 60, 120, 180, 240])
plt.title(f"Solitary Wave (skewness = {r_2})")
plt.savefig(f"./Graph/solitary_waves_{r_2}.png")

# Plot 2D
# plt.figure(dpi=300)
# for i in [240, 180, 120, 60, 1]:
#     plt.plot(alpha, xpoints[:, int(i / h - 1)], label=f"t={i}")
# plt.xlabel(r"$\alpha$")
# plt.ylabel(r"$x_{\alpha}$")
# plt.title(f"Solitary Wave (skewness = {r_2})")
# plt.legend()
# plt.grid()
# plt.savefig(f"./Graph/solitary_waves_{r_2}.png")
# plt.show()
