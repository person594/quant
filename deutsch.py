from quant import Qubit, measure, apply_operator, hadamard, sigmaX

import random
import numpy as np


def quantize(f, n_bits):
    a = np.zeros((2,) * (2*n_bits+2))
    for x in range(2**n_bits):
        index = tuple(int(bool(x & 2**i)) for i in range(n_bits))
        if f(x):
            a[index][1][index][0] = 1
            a[index][0][index][1] = 1
        else:
            a[index][1][index][1] = 1
            a[index][0][index][0] = 1
    return a

def constant0(x):
    return False

def constant1(x):
    return True

def gen_balanced(n_bits):
    l = [False] * 2**(n_bits-1) + [True] * 2**(n_bits-1)
    random.shuffle(l)
    return lambda x: l[x]

n_bits = 10

for trial in range(10):
    balanced = gen_balanced(n_bits)
    function = random.choice([constant0, constant1, balanced, balanced])
    if function == balanced:
        print('Oracle: Balanced')
    else:
        print('Oracle: Constant')
    a = quantize(function, n_bits)
    
    bits = [Qubit() for _ in range(n_bits)]
    
    ancilla = Qubit()
    sigmaX(ancilla)
    
    for bi in bits:
        hadamard(bi)
    hadamard(ancilla)

    apply_operator(a, *(bits + [ancilla]))
        
    for bi in bits:
        hadamard(bi)
    
    disjunction = False
    for bi in bits:
        disjunction = disjunction or measure(bi)

    if (disjunction):
        print('Predicted: Balanced')
        assert function == balanced
    else:
        print('Predicted: Constant')
        assert function != balanced
    print()
