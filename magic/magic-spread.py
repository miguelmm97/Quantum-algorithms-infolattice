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
#%% Parameters

# Initial state
psi0_label = '0' * 3
n_qubits = len(psi0_label)
psi1 = Statevector.from_label(psi0_label)
psi2 = Statevector.from_label(psi0_label)

# Circuit parameters
Nlayers = 30
depth = 20
seed_list = np.random.randint(0, 1000000, size=(Nlayers, ))
qubits = list(range(n_qubits))
T_per_layer = 2
max_layer = 10

# Preallocation
info_dict = {}
info_dict_clifford = {}


#%% Circuit

clifford_sequence = QuantumCircuit(n_qubits)
for i in range(Nlayers):

    # Generate Clifford layer
    layer = QuantumCircuit(n_qubits)
    clifford = random_clifford_circuit(num_qubits=n_qubits, depth=depth, seed=seed_list[i])
    clifford_sequence.compose(clifford, inplace=True)
    layer.compose(clifford, inplace=True)

    # Application of T gates
    rng = np.random.default_rng(seed_list[i])
    rng.shuffle(qubits)
    operands = qubits[-T_per_layer:]
    if i < max_layer:
        for qubit in operands:
            layer.t(qubit)

    # Information lattice
    psi1 = psi1.evolve(layer)
    psi2 = psi2.evolve(clifford)
    info_dict[i] = calc_info(psi1.data)
    info_dict_clifford[i] = calc_info(psi2.data)
    print(f'Layer: {i}, Info per scale |psi>:', calc_info_per_scale(info_dict[i], bc='open'))



#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']


Nrows = int(Nlayers / 10) + 1
Ncol = 10
fig1 = plt.figure(figsize=(20, 10))
fig1.suptitle('$\mathcal{U}\\vert \psi \\rangle$ with $\mathcal{U}=\langle H, S, CNOT, T\\rangle$', fontsize=20)
gs = GridSpec(Nrows, Ncol, figure=fig1, hspace=0, wspace=0.1)
for i in range(Nlayers):
    # Position in the grid
    row = i // Ncol
    col = i % Ncol
    ax = fig1.add_subplot(gs[row, col])
    # Plots
    plot_info_latt(info_dict[i], ax)
    if i < max_layer:
        ax.set_title(f'$n_l$: {i}, ' + '$T+\mathcal{C}$')
    else:
        ax.set_title(f'$n_l$: {i}, ' + '$\mathcal{C}$')


fig2 = plt.figure(figsize=(20, 10))
fig2.suptitle('$\mathcal{U}\\vert \psi \\rangle$ with $\mathcal{U}=\langle H, S, CNOT\\rangle$', fontsize=20)
gs = GridSpec(Nrows, Ncol, figure=fig2, hspace=0, wspace=0.1)
for i in range(Nlayers):
    # Position in the grid
    row = i // Ncol
    col = i % Ncol
    ax = fig2.add_subplot(gs[row, col])
    # Plots
    plot_info_latt(info_dict_clifford[i], ax)
    ax.set_title(f'$n_l$: {i}, ' + '$\mathcal{C}$')


# clifford_sequence.draw(output="mpl", style="iqp")
# clifford_sequence.inverse().draw(output="mpl", style="iqp")
plt.show()
