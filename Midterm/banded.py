#!/usr/local/bin/python3.11
# -*- coding: UTF-8 -*-
# @Project : Computational_Physics
# @File    : banded.py
# @Author  : Albert Wang
# @Time    : 2024/4/14
# @Brief   : None

import numpy as np


def qr_decomposition(A):
    """
    QR decomposition of a tridiagonal matrix A.

    :param A: a tridiagonal matrix.
    :return: the result of the decomposition. i.e Q, R.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError('Input Matrix must be square')  # Error handle

    dimension = A.shape[0]
    A_ = np.copy(A)  # Separate the following operations from the original matrix A
    Q_ = np.zeros_like(A_)  # Initialize Q
    R_ = np.zeros_like(A_)  # Initialize R

    for i in range(dimension):
        # Gram-Schmidt orthogonality
        u_ = np.copy(A_[:, i])
        for j in range(i):
            dot = 0
            # Only calculate the non-zero elements
            if i != 0 and i != dimension - 1:
                for k in range(3):
                    dot += Q_[i - 1 + k, j] * A_[i - 1 + k, i]
            # For the first and the last columns, there are only two non-zero elements
            elif i == 0:
                for k in range(2):
                    dot += Q_[k, j] * A_[k, i]
            elif i == dimension - 1:
                for k in range(2):
                    dot += Q_[dimension - 1 - k, j] * A_[dimension - 1 - k, i]
            u_ -= dot * Q_[:, j]
        Q_[:, i] = u_ / np.linalg.norm(u_)

        # Get R
        R_[i, i] = np.linalg.norm(u_)
        for j in range(i):
            dot = 0
            if i != 0 and i != dimension - 1:
                for k in range(3):
                    dot += Q_[i - 1 + k, j] * A_[i - 1 + k, i]
            elif i == 0:
                for k in range(2):
                    dot += Q_[k, j] * A_[k, i]
            elif i == dimension - 1:
                for k in range(2):
                    dot += Q_[dimension - 1 - k, j] * A_[dimension - 1 - k, i]
            R_[j, i] = dot

    return Q_, R_
