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
from InfoLattice import calc_info, plot_info_latt, calc_info_per_scale
from functions  import random_clifford_circuit
#%% Evolving quantum states through the circuit

# Initial state
psi0_label = '0' * 4
n_qubits = len(psi0_label)
psi1 = Statevector.from_label(psi0_label)
psi2 = Statevector.from_label(psi0_label)

# Circuit parameters
Nlayers = 30
depth = 20
seed_list = np.random.randint(0, 1000000, size=(Nlayers, ))     #[44,55,22,33,99,9,78,654,321,234,543,1,4,8,45,26,45,90,777,555,21,12]

info_dict = {}
info_dict_Ns = {}

clifford_sequence = QuantumCircuit(n_qubits)
for i in range(Nlayers):
    # Generate Clifford layer
    layer = QuantumCircuit(n_qubits)
    clifford = random_clifford_circuit(num_qubits=n_qubits, depth=depth, seed=seed_list[i])
    layer.compose(clifford, inplace=True)
    clifford_sequence.compose(clifford, inplace=True)

    # Randomly apply T gates
    rng = np.random.default_rng(seed_list[i])
    qubits = list(range(n_qubits))
    rng.shuffle(qubits)
    operands = qubits[-2:]
    if i < int(Nlayers / 4):
        layer.t(operands[0])
    # layer.t(operands[1])
    # layer.r(np.pi/3, np.pi/7, operands[0])
    # layer.r(np.pi/3, np.pi/7, operands[1])
    # if i == 0:
    #     layer.t(0)

    # Information lattice
    psi1 = psi1.evolve(layer)
    psi2 = psi2.evolve(clifford_sequence)
    info_dict[i] = calc_info(psi1.data)
    info_dict_Ns[i] = calc_info(psi2.data)
    print(f'Layer: {i}, Info per scale |psi>:', calc_info_per_scale(info_dict[i], bc='open'))


    # clifford_sequence.draw(output="mpl", style="iqp")
    # clifford_sequence.inverse().draw(output="mpl", style="iqp")
    # plt.show()

# clifford_undo = QuantumCircuit(2)
# clifford_undo.h(0)
# clifford_undo.cx(0, 1)
psi_try = psi1.evolve(clifford_sequence.inverse())
info_latt_try = calc_info(psi_try.data)
#
# print('Info per scale |psi>:', calc_info_per_scale(info_dict[i], bc='open'))
# print('Info per scale C|psi>:', calc_info_per_scale(info_latt_try, bc='open'))


fig1 = plt.figure()
gs = GridSpec(2, Nlayers, figure=fig1)
for i in range(Nlayers):
    ax1 = fig1.add_subplot(gs[0, i])
    ax2 = fig1.add_subplot(gs[1, i])
    plot_info_latt(info_dict[i], ax1)
    plot_info_latt(info_dict_Ns[i], ax2)

fig2 = plt.figure()
ax = fig2.gca()
plot_info_latt(info_latt_try, ax)

plt.show()



