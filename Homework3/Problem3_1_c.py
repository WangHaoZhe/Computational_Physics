#!/usr/local/bin/python3.11
# -*- coding: UTF-8 -*-
# @Project : Computational_Physics
# @File    : Problem3_1_c.py
# @Author  : Albert Wang
# @Time    : 2024/4/3
# @Brief   : None

import numpy as np


def lu_decomposition(A):
    """
    LU decomposition of a matrix A.

    :param A: a square matrix.
    :return: the result of the decomposition and the permutation matrix. i.e L, U and P.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError('Input Matrix must be square')  # Error handle

    dimension = A.shape[0]
    U_ = np.copy(A)  # Separate the following operations from the original matrix A
    L_ = np.zeros_like(U_)  # Initialize L
    P_ = np.eye(dimension)  # Initialize P

    for i in range(dimension):
        max_ = np.argmax(U_[i:dimension, i]) + i  # Find the largest number in each column
        # Swap rows
        temp = np.copy(U_[i])
        U_[i] = U_[max_]
        U_[max_] = temp

        temp = np.copy(P_[i])
        P_[i] = P_[max_]
        P_[max_] = temp

        for j in range(i, dimension):
            L_[j, i] = U_[j, i]  # Assemble L

        U_[i] /= U_[i, i]  # Divide by the diagonal element
        for j in range(i + 1, dimension):
            U_[j, :] -= U_[j, i] * U_[i, :]  # Subtract from the lower rows

    return L_, U_, P_


def lu_solution(A, v):
    """
    Solve Ax=v.
    :param A: a square matrix.
    :param v: a vector which has the same dimension as A.
    :return: the solution vector.
    """
    L_, U_, P_ = lu_decomposition(A)

    dimension = A.shape[0]
    y_ = np.zeros(dimension)
    x_ = np.zeros(dimension)
    u_ = np.copy(v)
    u_ = np.dot(P_, u_)

    # Calculate vector y
    for i in range(dimension):
        y_[i] = u_[i]
        for j in range(i):
            y_[i] -= L_[i, j] * y_[j]
        y_[i] /= L_[i, i]

    # Calculate vector x
    for i in range(dimension):
        x_[dimension - i - 1] = y_[dimension - i - 1]
        for j in range(i):
            x_[dimension - i - 1] -= x_[dimension - j - 1] * U_[dimension - i - 1, dimension - j - 1]
        x_[dimension - i - 1] /= U_[dimension - i - 1, dimension - i - 1]

    return x_


if __name__ == '__main__':
    M = np.array([[0, 1, 4, 1],
                  [3, 4, -1, -1],
                  [1, -4, 1, 5],
                  [2, -2, 1, 3]], float)
    v = np.array([-4, 3, 9, 7])

    L, U, P = lu_decomposition(M)
    print("L:", L)
    print("U:", U)
    print("P:", P)
    print(np.dot(L, U))  # Verify

    x = lu_solution(M, v)
    print("x:", x)
    print(np.linalg.solve(M, v))  # Verify
