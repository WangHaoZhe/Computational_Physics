#!/usr/local/bin/python3.11
# -*- coding: UTF-8 -*-
# @Project : Computational_Physics
# @File    : mid_prob_1_3.py
# @Author  : Albert Wang
# @Time    : 2024/4/11
# @Brief   : None

import numpy as np

from matplotlib import pyplot as plt
from mid_prob_1_1 import chebyshev_sum_norm


def find_c(k_, sigma_):
    c_ = np.zeros(k_ + 1)
    c_[0] = 1
    for i in range(1, k_ + 1):
        c_[i] = 2 * np.cos(i * np.arccos(sigma_))

    return c_


def chebyshev_sum_pref(sigma_, k_, epsilon_):
    """
    Calculate the linear combination of Chebyshev polynomials based on the Clenshaw algorithm.

    :param sigma_: A constant parameter for the Chebyshev polynomials sum.
    :param k_: An integer represents the order of the Chebyshev polynomials sum
    :param epsilon_: An array of variables
    :return: The sum of the Chebyshev polynomials
    """
    c_ = find_c(k_, sigma_)
    N_ = c_.shape[0] - 1
    b_ = np.zeros((N_ + 1, epsilon_.shape[0]))
    b_[N_].fill(0)
    b_[N_ - 1].fill(c_[N_])
    for i in range(N_ - 1, 0, -1):
        b_[i - 1] = 2 * epsilon_ * b_[i] - b_[i + 1] + c_[i]  # Operate to the whole array

    return epsilon_ * b_[0] - b_[1] + c_[0]


def chebyshev_sum_pref_norm(sigma_, k_, epsilon_):
    p_ = chebyshev_sum_pref(sigma_, k_, epsilon_)
    norm_ = chebyshev_sum_pref(sigma_, k_, np.array([sigma_]))
    return p_ / norm_


if __name__ == '__main__':
    e = np.linspace(-1, 1, 200)
    plt.plot(e, chebyshev_sum_pref_norm(0, 22, e), color="#ff7f0e", label="New", marker=".")
    plt.plot(e, chebyshev_sum_norm(0, 22, e), color="#1f77b4", label="Old")
    plt.legend()
    plt.show()
