#!/usr/local/bin/python3.11
# -*- coding: UTF-8 -*-
# @Project : Computational_Physics
# @File    : final_prob_1_4.py
# @Author  : Albert Wang
# @Time    : 2024/6/7
# @Brief   : None

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy

# Constants
S = 13
r_1 = 1.0
r_2 = 0.5  # 0.5 for Problem 1.4, 1.0 for Problem 1.5, 2.0 for Problem 1.6
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


# Polarization
def simpson(x_, a_, b_, h_):
    return (np.sum(x_[1:-1:2] * 4 + x_[2:-1:2] * 2) + x_[0] + x_[-1]) * h_ / 3


x_average = np.zeros(S)
for alpha in range(S):
    x_average[alpha] = simpson(xpoints[alpha, :], a, b, h) / (b - a)
alpha = np.arange(1, S + 1, 1)


# Fit
def fit_func(x, a, b):
    if r_2 == 1.0:
        return a
    else:
        return a * np.exp(b * x)


popt, pcov = curve_fit(fit_func, alpha, x_average)

# Plot 2D
plt.figure(dpi=300)
plt.plot(alpha, x_average, label=r"$\langle x_{\alpha} \rangle _{T}$")
if r_2 == 1.0:
    plt.plot(
        alpha,
        popt[0] * np.ones_like(alpha),
        label="Fitted Curve($" + str(round(popt[0], 3)) + "$)",
    )
else:
    plt.plot(
        alpha,
        fit_func(alpha, *popt),
        label="Fitted Curve($"
        + str(round(popt[0], 3))
        + "\cdot e^{"
        + str(round(popt[1], 3))
        + "\cdot x}$)",
    )
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\langle x_{\alpha} \rangle _{T}$")
plt.title(f"Mass Polarization (skewness = {r_2})")
plt.legend()
plt.grid()
plt.savefig(f"./Graph/mass_polarization_{r_2}.png")
plt.show()

# zero_space = scipy.linalg.null_space(A)
# print("null_space:", zero_space)
