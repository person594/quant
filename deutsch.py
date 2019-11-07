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

def balanced(x):
    parity = False
    while x:
        x &= x-1
        parity = not parity
    return parity

    

for trial in range(20):
    function = random.choice([constant0, constant1, balanced])
    if function == balanced:
        print('Oracle: Balanced')
    else:
        print('Oracle: Constant')
    a = quantize(function, 3)
    
    b1 = Qubit()
    b2 = Qubit()
    b3 = Qubit()
    ba = Qubit()
    sigmaX(ba)
    
    hadamard(b1)
    hadamard(b2)
    hadamard(b3)
    hadamard(ba)

    apply_operator(a, b1, b2, b3, ba)
    
    hadamard(b1)
    hadamard(b2)
    hadamard(b3)
    
    if measure(b1):
        print('Predicted: Balanced')
    else:
        print('Predicted: Constant')
    print()
