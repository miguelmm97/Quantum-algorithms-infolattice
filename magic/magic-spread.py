#%% Imports

# Built-in modules
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# Information lattice
from InfoLattice import calc_info, plot_info_latt, calc_info_per_scale
from functions import random_clifford_circuit
#%% Parameters

# Initial state
psi0_label = '0' * 10
n_qubits = len(psi0_label)
psi1 = Statevector.from_label(psi0_label)
psi2 = Statevector.from_label(psi0_label)

# Circuit parameters
Nlayers = 30
depth = 10
seed_list = np.random.randint(0, 1000000, size=(Nlayers, ))
qubits = list(range(n_qubits))
T_per_layer = 2
max_layer = 1

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
    # print(f'Layer: {i}, Info per scale |psi>:', calc_info_per_scale(info_dict[i], bc='open'))



#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']

# Defining a colormap (starting in white)
color_map = plt.get_cmap("Oranges")
colors = color_map(np.linspace(0, 1, 20))
colors[0] = [1, 1, 1, 1]
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)

# Normalising the colormap
max_value = 0.
for key1 in info_dict_clifford.keys():
    for key2 in info_dict_clifford[key1].keys():
        value = max(info_dict_clifford[key1][key2])
        if value > max_value:
            max_value = value
colormap = cm.ScalarMappable(norm=Normalize(vmin=0., vmax=max_value), cmap=color_map)


# Fig 1: Plot
Nrows = int(Nlayers / 10)
Ncol = 10
fig1 = plt.figure(figsize=(20, 10))
fig1.suptitle('$\mathcal{U}\\vert \psi \\rangle$ with $\mathcal{U}=\langle H, S, CNOT, T\\rangle$', fontsize=20)
gs = GridSpec(Nrows, Ncol+1, figure=fig1, hspace=0, wspace=0.1)
for i in range(Nlayers):
    # Position in the grid
    row = i // Ncol
    col = i % Ncol
    ax = fig1.add_subplot(gs[row, col])
    # Plots
    plot_info_latt(info_dict[i], ax, color_map, max_value=max_value, indicate_ints=True)
    if i < max_layer:
        ax.set_title(f'$n_l$: {i}, ' + '$T+\mathcal{C}$')
    else:
        ax.set_title(f'$n_l$: {i}, ' + '$\mathcal{C}$')

# Fig 1: Colorbar
cbar_ax = fig1.add_subplot(gs[1, -1])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("left", size="20%", pad=0)
cbar = fig1.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()
cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)


# Fig 2: Plots
fig2 = plt.figure(figsize=(20, 10))
fig2.suptitle('$\mathcal{U}\\vert \psi \\rangle$ with $\mathcal{U}=\langle H, S, CNOT\\rangle$', fontsize=20)
gs = GridSpec(Nrows, Ncol + 1, figure=fig2, hspace=0, wspace=0.1)
for i in range(Nlayers):
    # Position in the grid
    row = i // Ncol
    col = i % Ncol
    ax = fig2.add_subplot(gs[row, col])
    # Plots
    plot_info_latt(info_dict_clifford[i], ax, color_map, max_value=max_value, indicate_ints=True)
    ax.set_title(f'$n_l$: {i}, ' + '$\mathcal{C}$')

# Fig 2: Colorbar
cbar_ax = fig2.add_subplot(gs[1, -1])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("left", size="20%", pad=0)
cbar = fig2.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()
cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)

# clifford_sequence.draw(output="mpl", style="iqp")
plt.show()
