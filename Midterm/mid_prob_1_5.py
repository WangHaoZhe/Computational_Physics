#!/usr/local/bin/python3.11
# -*- coding: UTF-8 -*-
# @Project : Computational_Physics
# @File    : mid_prob_1_5.py
# @Author  : Albert Wang
# @Time    : 2024/4/12
# @Brief   : None

import numpy as np
import scipy


def find_c(k_, sigma_):
    c_ = np.zeros(k_ + 1)
    c_[0] = 1
    for i in range(1, k_ + 1):
        c_[i] = 2 * np.cos(i * np.arccos(sigma_))

    return c_


def chebyshev_polynomial(A, n):
    """
    Chebyshev polynomial of a matrix A.

    :param A: a square matrix.
    :param n: the order of the Chebyshev polynomial.
    :return: the Chebyshev polynomial result of A.
    """
    I = np.eye(A.shape[0])
    if n == 0:
        return I
    elif n == 1:
        return A
    else:
        T_n_minus_2 = I
        T_n_minus_1 = A
        for _ in range(n - 1):
            T_n = 2 * A.dot(T_n_minus_1) - T_n_minus_2
            T_n_minus_2, T_n_minus_1 = T_n_minus_1, T_n
        return T_n


def chebyshev_sum(sigma_, k_, epsilon_):
    c_ = find_c(k_, sigma_)
    sum_ = 0
    for i in range(k_ + 1):
        sum_ += c_[i] * chebyshev_polynomial(epsilon_, i)
    return sum_


if __name__ == '__main__':
    psi = np.loadtxt('random_vector.txt', float)
    central_diag = np.loadtxt("central_diag.txt", float)
    up_diag = np.loadtxt("up_diag.txt", float)

    # Construct and normalize H
    H = scipy.sparse.diags([up_diag, central_diag, up_diag], [-1, 0, 1])
    e_max = 21.79048303
    e_min = -21.79048303
    H_normalized = (2 * H - (e_max + e_min) * np.eye(H.shape[0])) / (e_max - e_min)

    # Calculate the normalization constant D
    P_norm = chebyshev_sum(0, 22, np.array([0]))
    P = chebyshev_sum(0, 22, H_normalized) / P_norm

    # Calculate the normalized psi
    psi_normalized = psi / np.sqrt(np.dot(psi, psi))

    # Calculate the final result
    result = np.dot(psi_normalized.T, np.dot(P, psi_normalized))
    print(result)
