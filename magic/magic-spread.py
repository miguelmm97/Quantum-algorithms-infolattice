#%% Imports

# Built-in modules
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, HGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator

# Information lattice
from InfoLattice import calc_info, plot_info_latt
from functions  import random_clifford_circuit
#%% Evolving quantum states through the circuit

# Initial state
psi0_label = '000000'
n_qubits = len(psi0_label)
psi = Statevector.from_label(psi0_label)

# Circuit parameters
Nlayers = 10
depth = 20
seed_list =[1,4,8,45,26,45,90,777,555,21,12]
print(seed_list)
info_dict = {}
info_dict_Ns = {}

for i in range(Nlayers):
    # Generate Clifford layer
    layer = QuantumCircuit(n_qubits)
    clifford_sequence = QuantumCircuit(n_qubits)
    inv_clifford_sequence = QuantumCircuit(n_qubits)
    clifford = random_clifford_circuit(num_qubits=n_qubits, depth=depth, seed=seed_list[i])
    layer.compose(clifford, inplace=True)
    clifford_sequence.compose(clifford, inplace=True)
    inv_clifford_sequence.compose(clifford.inverse(), inplace=True)

    # Randomly apply T gates
    rng = np.random.default_rng(seed_list[i])
    qubits = list(range(n_qubits))
    rng.shuffle(qubits)
    operands = qubits[-2:]
    # layer.t(operands[0])
    # layer.t(operands[1])

    # Information lattice
    psi = psi.evolve(layer)
    psi_Ns = psi.evolve(inv_clifford_sequence.inverse())
    info_dict[i] = calc_info(psi.data)
    info_dict_Ns[i] = calc_info(psi_Ns.data)

    # clifford_sequence.draw(output="mpl", style="iqp")
    # clifford_sequence.inverse().draw(output="mpl", style="iqp")
    plt.show()




fig1 = plt.figure(figsize=(10, 7))
gs = GridSpec(2, Nlayers, figure=fig1)
for i in range(Nlayers):
    ax1 = fig1.add_subplot(gs[0, i])
    ax2 = fig1.add_subplot(gs[1, i])
    plot_info_latt(info_dict[i], ax1)
    plot_info_latt(info_dict_Ns[i], ax2)
plt.show()



