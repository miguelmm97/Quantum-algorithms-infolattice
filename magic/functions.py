import numpy as np
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.circuit.library.standard_gates import HGate, SGate, TGate, CXGate
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
from itertools import combinations_with_replacement, product




def random_clifford_circuit(num_qubits, depth, max_operands=2, seed=None):
    """Generate random circuit of arbitrary size and form.
    This function will generate a random circuit by randomly selecting gates
    from the set of standard gates in :mod:`qiskit.extensions`. For example:
    .. jupyter-execute::
        from qiskit.circuit.random import random_circuit
        circ = random_circuit(2, 2, measure=True)
        circ.draw(output='mpl')
    Args:
        num_qubits (int): number of quantum wires
        depth (int): layers of operations (i.e. critical path length)
        max_operands (int): maximum operands of each gate (between 1 and 3)
        measure (bool): if True, measure all qubits at the end
        conditional (bool): if True, insert middle measurements and conditionals
        reset (bool): if True, insert middle resets
        seed (int): sets random seed (optional)
    Returns:
        QuantumCircuit: constructed circuit
    Raises:
        CircuitError: when invalid options given
    """
    if max_operands < 1 or max_operands > 3:
        raise CircuitError("max_operands must be between 1 and 3")

    # List of Clifford gates
    one_q_ops = [HGate, SGate]
    two_q_ops = [CXGate]

    # Initialise circuit and seed
    qr = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(num_qubits)
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    # Apply arbitrary random operations at every depth
    for _ in range(depth):

        # Choose either 1, 2 qubits for the operation
        remaining_qubits = list(range(num_qubits))
        while remaining_qubits:

            # Seeing if there are possible neighbour pairs remaining
            diff = 100
            allow_2qubit = False
            for q1 in remaining_qubits:
                for q2 in remaining_qubits:
                    diff = np.abs(q1 - q2)
                    if diff == 1:
                        allow_2qubit = True
                        break
                if diff == 1:
                    break

            # Selecting number of qubits to apply the gate
            if allow_2qubit:
                max_possible_operands = min(len(remaining_qubits), max_operands)
                num_operands = rng.choice(range(max_possible_operands)) + 1
            else:
                num_operands = 1

            # Selecting to which qubit(s) we apply the gate
            if num_operands == 1:
                rng.shuffle(remaining_qubits)
                operands = [remaining_qubits.pop() for _ in range(num_operands)]
                operation = rng.choice(one_q_ops)
            elif num_operands == 2:
                diff = 100
                while diff != 1:
                    rng.shuffle(remaining_qubits)
                    operands = remaining_qubits[-2:]
                    diff = np.abs(operands[0] - operands[1])
                operands = [remaining_qubits.pop() for _ in range(num_operands)]
                operation = rng.choice(two_q_ops)

            # Update quantum circuit
            register_operands = [qr[i] for i in operands]
            qc.append(operation(), register_operands)
        qc.barrier()
    return qc


def stabiliser_Renyi_entropy_pure(state, n, num_qubits):

    # Pauli strings
    pauli_strings = []
    pauli_iter = product('IXYZ', repeat=num_qubits)
    for element in pauli_iter:
        pauli_strings.append(''.join(element))

    # Probability distribution
    rho = DensityMatrix(state).data
    prob_dist = np.zeros((len(pauli_strings), ))
    for i, pauli_string in enumerate(pauli_strings):
        P = Pauli(pauli_string).to_matrix()
        prob_dist[i] = (np.abs(np.trace(rho @ P)) ** 2) / (2 ** num_qubits)

    # Renyi entropy
    Mn_rho = (1 / (1 - n)) * np.log(np.sum(prob_dist ** n)) - np.log(2 ** num_qubits)
    return Mn_rho

def stabiliser_Renyi_entropy_mixed(rho, num_qubits):

    # Pauli strings
    pauli_strings = []
    pauli_iter = product('IXYZ', repeat=num_qubits)
    for element in pauli_iter:
        pauli_strings.append(''.join(element))

    # Probability distribution
    Tr_rhoP = np.zeros((len(pauli_strings),))
    for i, pauli_string in enumerate(pauli_strings):
        P = Pauli(pauli_string).to_matrix()
        Tr_rhoP[i] = np.abs(np.trace(rho @ P))

    # Renyi entropy
    M2_rho = - np.log(np.sum(Tr_rhoP ** 4) / np.sum(Tr_rhoP ** 2))
    return M2_rho


def kron_iter(L, up_position=0):
    """ Given a position returns |00...1...000...> in that position."""
    if L == 1:
        if up_position == L-1:
            return spinup()
        else:
            return spindown()
    else:
        if up_position == L-1:
            return np.kron(kron_iter(L - 1, up_position), spinup())
        else:
            return np.kron(kron_iter(L - 1, up_position), spindown())

def w_state_n(n):
    """ Creates a W-state of order n: |100...> + |010...> + ... """
    state = np.zeros(2**n, dtype=float)
    for i in range(n):
        state += kron_iter(n, i)
    state /= np.sqrt(n)
    return state

def wn_in_chain(L, n):
    """ The first n elements of this chain are W-like entangled."""
    assert L >= n
    if L == n:
        return w_state_n(n)
    else:
        return np.kron(wn_in_chain(L - 1, n), spindown())

def random_wn_state(n, k):
    order = np.random.default_rng().permutation(n)
    L = len(order)
    psi = wn_in_chain(L, k)
    psi = np.transpose(np.reshape(psi, [2] * L), axes=np.argsort(order))
    return psi.flatten()

def spinup():
    return np.array([1, 0])

def spindown():
    return np.array([0, 1])








