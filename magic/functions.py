import numpy as np
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Reset
from qiskit.circuit.library.standard_gates import HGate, SGate, TGate, CXGate
from qiskit.circuit.exceptions import CircuitError




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











