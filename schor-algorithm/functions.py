import numpy as np
from scipy.linalg import eigvalsh as scipy_eigvalsh
from numpy.linalg import eigvalsh as numpy_eigvalsh
import matplotlib.pyplot as plt
from numpy import pi

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, HGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator

# Managing data
import h5py
import os



# Algorithm
def qft_circuit(num_qubits):
    circuit = QuantumCircuit(num_qubits)
    for n in range(num_qubits - 1, -1, -1):
        if n == 0:
            circuit.h(0)
            return circuit
        else:
            qft_block(circuit, n)
            circuit.barrier()
    return circuit

def qft_block(circuit, qubit):
    circuit.h(qubit)
    for target in range(qubit - 1, -1, - 1):
        print(qubit, target)
        circuit.cp(pi / 2 ** (qubit - target), target, qubit)