import matplotlib.pyplot as plt
import numpy as np
from quant import Qubit, hadamard, cnot, rotate, measure

def correlation(theta, n=1000):
    sm = 0
    for trial in range(n):
        b1 = Qubit()
        b2 = Qubit()
        hadamard(b1)
        cnot(b1, b2)
        rotate(theta, b1)
        v1 = 2*measure(b1) - 1
        v2 = 2*measure(b2) - 1
        sm += v1*v2
    return sm / n

thetas = np.arange(0, 2*np.pi, np.pi/16)
correlations = []
for theta in thetas:
    correlations.append(correlation(theta))
plt.plot(thetas, correlations)
plt.show()
