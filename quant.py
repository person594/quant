import numpy as np

def iter_bitstrings(n_bits):
    if n_bits == 0:
        yield ()
    else:
        for bitstring in iter_bitstrings(n_bits-1):
            yield bitstring + (0,)
            yield bitstring + (1,)

class Qubit:
    def __init__(self):
        self.space = HilbertSpace([self])

    def density_matrix(self):
        i = self.space.qubits.index(self)
        bits_before = i
        bits_after = len(self.space.qubits) - i - 1
        state = self.space.state
        rho = np.zeros((2,2))
        for prefix in iter_bitstrings(bits_before):
            for suffix in iter_bitstrings(bits_after):
                rho[0,0] += state[prefix + (0,) + suffix] **2
                rho[0,1] += state[prefix + (0,) + suffix] * state[prefix + (1,) + suffix]
                rho[1,1] += state[prefix + (1,) + suffix] **2
        rho[1,0] = rho[0,1]
        return rho
    
    def __repr__(self):
        rho = self.density_matrix()
        return str(rho)

class HilbertSpace:
    def __init__(self, qubits = []):
        self.qubits = list(qubits)
        self.state = np.zeros([2] * len(self.qubits))
        self.state[(0,) * len(self.qubits)] = 1
    
    def entangle(self, other):
        if other is self:
            return self
        new_space = HilbertSpace(self.qubits + other.qubits)
        new_space.state = np.tensordot(self.state, other.state, 0)
        for qubit in self.qubits:
            qubit.space = new_space
        for qubit in other.qubits:
            qubit.space = new_space
        return new_space

    # TODO: detect if the space can be rewritten as a tensor product of
    # unentangled states
    # In the general case, this is NP-hard (https://en.wikipedia.org/wiki/Tensor_rank_decomposition),
    # but it might be possible for some special cases
    def disentangle(self):
        return [self]

def apply_operator(operator, *qubits):
    space = None
    assert len(set(qubits)) == len(qubits)
    for qubit in qubits:
        if space is None:
            space = qubit.space
        else:
            space = space.entangle(qubit.space)
    # transpose our tensor so our qubits are at the back, apply the operator,
    # and then transpose back
    indices = [space.qubits.index(qubit) for qubit in qubits]
    transposition = [i for i in range(len(space.qubits)) if i not in indices] + indices
    inverse_transposition = [transposition.index(i) for i in range(len(space.qubits))]
    state = space.state.transpose(transposition)
    state = np.tensordot(state, operator, len(qubits))
    space.state = state.transpose(inverse_transposition)
    space.disentangle()


def measure(b):
    space = b.space
    state = space.state
    i = b.space.qubits.index(b)
    index = [slice(None) for _ in space.qubits]
    axes = list(range(len(space.qubits)))
    del axes[i]
    axes = tuple(axes)
    probs = np.sum(np.abs(state)**2, axis=axes)
    del space.qubits[i]
    b.space = HilbertSpace([b])
    if np.random.random() < probs[0]:
        index[i] = 0
        result = 0
    else:
        index[i] = 1
        b.space.state = np.array([0, 1])
        result = 1
    index = tuple(index)
    space.state = space.state[index]
    space.state = space.state / np.sqrt(np.sum(np.abs(space.state)**2))
    return result
    

def sigmaX(b):
    operator = np.array([[0, 1], [1, 0]])
    apply_operator(operator, b)

def sigmaY(b):
    operator = np.array([[0, 1j], [-1j, 0]])
    apply_operator(operator, b)

def sigmaZ(b):
    operator = np.array([[1, 0], [0, -1]])
    apply_operator(operator, b)


def hadamard(b):
    operator = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    apply_operator(operator, b)

def cnot(b1, b2):
    operator = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])
    operator.shape = (2, 2, 2, 2)
    
    apply_operator(operator, b1, b2)

def toffoli(b1, b2, b3):
    operator = np.eye(8)
    operator[6,6] = 0
    operator[7,7] = 0
    operator[6,7] = 1
    operator[7,6] = 1
    operator.shape = (2, 2, 2, 2, 2, 2)
    apply_operator(operator, b1, b2, b3)

def rotate(theta, b):
    operator = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    apply_operator(operator, b)
