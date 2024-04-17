#!/usr/local/bin/python3.11
# -*- coding: UTF-8 -*-
# @Project : Computational_Physics
# @File    : mid_prob_1_4.py
# @Author  : Albert Wang
# @Time    : 2024/4/11
# @Brief   : None

import numpy as np
import banded


def qreigen(A, num):
    for i in range(num):
        v = np.diag(A)
        q, r = banded.qr_decomposition(A)
        A = np.dot(r, q)
        tol = max(np.diag(A)) - max(v)
        if np.abs(tol) < 1e-6:
            break

    return np.diag(A)


if __name__ == '__main__':
    central_diag = np.loadtxt("central_diag.txt", float)
    up_diag = np.loadtxt("up_diag.txt", float)
    H = np.diag(central_diag) + np.diag(up_diag, k=1) + np.diag(up_diag, k=-1)

    sol = np.linalg.eigvals(H)
    sol.sort()
    print(sol)

    s = qreigen(H, 200)
    s = np.sort(s)
    print(s)
