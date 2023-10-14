from math import *


def coefficient2value(ppolynomial, w=None):
    length = len(ppolynomial)
    if length == 1: return ppolynomial
    if w is None: w = complex(
        cos(2 * pi / length), 
        sin(2 * pi / length))

    peven = ppolynomial[::2]
    podd = ppolynomial[1::2]

    fodd = coefficient2value(podd, w**2)
    feven = coefficient2value(peven, w**2)
    fpolynomial, x = [0] * length, 1
    for i in range(length // 2):
        fpolynomial[i] = feven[i] + x * fodd[i]
        fpolynomial[i + length // 2] = feven[i] - x * fodd[i]
        x = x * w

    return fpolynomial


def value2coefficient(ppolynomial, w=None):
    length = len(ppolynomial)
    if length == 1: return ppolynomial
    if w is None: w = complex(
        cos(-2 * pi / length), 
        sin(-2 * pi / length))

    peven = ppolynomial[::2]
    podd = ppolynomial[1::2]

    fodd = coefficient2value(podd, w**2)
    feven = coefficient2value(peven, w**2)
    fpolynomial, x = [0] * length, 1
    for i in range(length // 2):
        fpolynomial[i] = (feven[i] + x * fodd[i]) / length
        fpolynomial[i + length // 2] = (feven[i] - x * fodd[i]) / length
        x = x * w

    return fpolynomial


def multiply(A: list, B: list):
    length = 2**ceil(log2(len(A) + len(B)))
    for i in range(len(A), length): A.append(0)
    for i in range(len(B), length): B.append(0)
    values1 = coefficient2value(A)
    values2 = coefficient2value(B)
    C = [values1[i] * values2[i] for i in range(length)]
    C = [i.real for i in value2coefficient(C)]
    return C



