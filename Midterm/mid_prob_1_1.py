#!/usr/local/bin/python3.11
# -*- coding: UTF-8 -*-
# @Project : Computational_Physics
# @File    : mid_prob_1_1.py
# @Author  : Albert Wang
# @Time    : 2024/4/11
# @Brief   : None

import numpy as np

from matplotlib import pyplot as plt


def chebyshev_t(n, x):
    return np.cos(n * np.arccos(x))


def chebyshev_sum(sigma_, k_, epsilon_):
    """
    Calculate the linear combination of Chebyshev polynomials

    :param sigma_: A constant parameter for the Chebyshev polynomials sum.
    :param k_: An integer represents the order of the Chebyshev polynomials sum
    :param epsilon_: An array of variables
    :return: The sum of the Chebyshev polynomials
    """
    sum_ = np.ones(epsilon_.shape[0])  # n=0

    for i in range(1, k_ + 1):
        sum_ += 2 * np.cos(i * np.arccos(sigma_)) * chebyshev_t(i, epsilon_)

    return sum_


def chebyshev_sum_norm(sigma_, k_, epsilon_):
    p_ = chebyshev_sum(sigma_, k_, epsilon_)
    norm_ = chebyshev_sum(sigma_, k_, np.array([sigma_]))
    return p_ / norm_


if __name__ == '__main__':
    e = np.linspace(-1, 1, 200)
    y = chebyshev_sum_norm(0, 22, e)
    plt.plot(e, y)
    plt.show()
