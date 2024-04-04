#!/usr/local/bin/python3.11
# -*- coding: UTF-8 -*-
# @Project : Computational_Physics
# @File    : Problem3_2.py
# @Author  : Albert Wang
# @Time    : 2024/4/3
# @Brief   : None

import numpy as np
import scipy


def qr_decomposition(A):
    """
    QR decomposition of a matrix A.

    :param A: a square matrix.
    :return: the result of the decomposition. i.e Q, R.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError('Input Matrix must be square')  # Error handle

    dimension = A.shape[0]
    A_ = np.copy(A)  # Separate the following operations from the original matrix A
    Q_ = np.zeros_like(A_)  # Initialize Q
    R_ = np.zeros_like(A_)  # Initialize R

    for i in range(dimension):
        # Gram-Schmidt Orthogonalization
        u_ = np.copy(A_[:, i])
        for j in range(i):
            u_ -= np.dot(Q_[:, j], A_[:, i]) * Q_[:, j]
        Q_[:, i] = u_ / np.linalg.norm(u_)

        # Get R
        R_[i, i] = np.linalg.norm(u_)
        for j in range(i):
            R_[j, i] = np.dot(Q_[:, j], A_[:, i])

    return Q_, R_


def qr_eigens(A, max_iter):
    """
    Get eigenvalues and eigenvectors of a matrix A by solving QR decomposition.

    :param A: a square matrix.
    :param max_iter: max iteration number
    :return: An array contains the eigenvalues and a matrix contains eigenvectors.
    """
    dimension = A.shape[0]
    A_ = np.copy(A)  # Separate the following operations from the original matrix A
    eigenvector_ = np.zeros_like(A_)

    for i in range(int(max_iter)):
        Q_, R_ = qr_decomposition(A_)
        A_ = np.dot(R_, Q_)

        # Get the max off-diag element
        off_diagonal = []
        for j in range(1, dimension):
            off_diagonal.append(np.max(np.diag(A_, k=j)))

        # Stop the iteration when the off-diagonal elements are smaller than 1e-6
        if max(off_diagonal) < 1e-6:
            eigenvalue_ = np.diag(A_)
            # Calculate the corresponding eigenvector
            for j in range(dimension):
                K_ = A - eigenvalue_[j] * np.eye(dimension)
                null_space = scipy.linalg.null_space(K_)
                for k in range(null_space.shape[0]):
                    eigenvector_[k, j] = null_space[k, 0]
            return eigenvalue_, eigenvector_

    raise ValueError("QR iteration did not converge")  # Not converge handle


if __name__ == '__main__':
    M = np.array([[1, 4, 8, 4],
                  [4, 2, 3, 7],
                  [8, 3, 6, 9],
                  [4, 7, 9, 2]], float)
    Q, R = qr_decomposition(M)
    print("Q:", Q)
    print("R:", R)
    print(np.dot(Q, R))  # Verify

    eigenvalue, eigenvector = qr_eigens(M, 1e6)
    print("Eigenvalues:", eigenvalue)
    print("Eigenvectors:", eigenvector)
