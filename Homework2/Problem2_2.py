#!/usr/local/bin/python3.11
# -*- coding: UTF-8 -*-
# @Project : Computational_Physics
# @File    : Problem2_2.py
# @Author  : Albert Wang
# @Time    : 2024/3/17
# @Brief   : None

from numpy import tanh, linspace
from matplotlib import pyplot as plt


def f(x):
    return 1 + 0.5 * tanh(2 * x)


def f_first_derivative(x):
    return 1 - tanh(2 * x) ** 2  # The analytic derivative of f(x)


def central_difference(x):
    h = 1e-3  # Set h
    return (f(x + h / 2) - f(x - h / 2)) / h


if __name__ == '__main__':
    x_ = linspace(-2, 2, 201)
    plt.plot(x_, central_difference(x_), color="#1f77b4", label="Numerical Solution",
             marker=".")  # Plot the numerical solution as dots
    plt.plot(x_, f_first_derivative(x_), color="#ff7f0e", label="Analytical Solution")
    plt.legend()
    plt.show()
