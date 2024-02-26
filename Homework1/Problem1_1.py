#!/usr/local/bin/python3.11
# -*- coding: UTF-8 -*-
# @Project : Computational_Physics
# @File    : Problem1_1.py
# @Author  : Albert Wang
# @Time    : 2024/2/26
# @Brief   : None

from math import sqrt


def isPrimeNumber(num):
    max_check = int(sqrt(num))

    for i in range(2, max_check + 1):  # Check whether the num is divisible by 2~sqrt(n)
        if num % i == 0:
            return False
    return True


primes = [2]
for i in range(3, 10001):
    if isPrimeNumber(i):
        primes.append(i)  # If is prime then append it to the list
print(primes)
