#%% Imports

# Built-in modules
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Managing data
import os
import sys
import h5py
from datetime import date

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace

# Information lattice and functions
from modules.InfoLattice import calc_info, plot_info_latt
from modules.MagicLattice import calc_magic, calc_classical_magic, calc_total_info, shannon, plot_magic_latt, nothing
from modules.functions import *


#%% Logging setup
loger_main = logging.getLogger('main')
loger_main.setLevel(logging.INFO)

stream_handler = colorlog.StreamHandler()
formatter = ColoredFormatter(
    '%(black)s%(asctime) -5s| %(blue)s%(name) -10s %(black)s| %(cyan)s %(funcName) '
    '-40s %(black)s|''%(log_color)s%(levelname) -10s | %(message)s',
    datefmt=None,
    reset=True,
    log_colors={
        'TRACE': 'black',
        'DEBUG': 'purple',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

stream_handler.setFormatter(formatter)
loger_main.addHandler(stream_handler)

#%% Parameters

# Initial state
n_qubits = 6
psi0_label = '0' * n_qubits
psi1 = Statevector.from_label(psi0_label)
psi2 = Statevector.from_label(psi0_label)

# Circuit parameters
Ncliff = 200
nT = 3
seed_list = np.random.randint(0, 1000000, size=(Ncliff, ))
qubits = list(range(n_qubits))

# Preallocation
info_dict = {}
SRE1_latt_dict = {}
magic_latt_dict = {}
nonint_magic = np.zeros((2, ))
total_info = np.zeros((2, ))

#%% Circuit

clifford_sequence = QuantumCircuit(n_qubits)
magic_circuit = QuantumCircuit(n_qubits)
for i in range(Ncliff):
    clifford = random_clifford_circuit(num_qubits=n_qubits, depth=1, seed=seed_list[i])
    clifford_sequence.compose(clifford, inplace=True)


rng = np.random.default_rng(seed_list[0])
rng.shuffle(qubits)
operands = qubits[-nT:]
print(operands)

layer = QuantumCircuit(n_qubits)
for qubit in operands:
    layer.t(qubit)
    # magic_circuit.t(qubit)

magic_circuit.compose(clifford_sequence, inplace=True)
magic_circuit.compose(layer, inplace=True)


psi_clifford = psi1.evolve(clifford_sequence)
psi_magic = psi1.evolve(magic_circuit)

info_dict[0] = calc_info(psi_clifford.data)
info_dict[1] = calc_info(psi_magic.data)

SRE1_latt_dict[0] = calc_classical_magic(psi_clifford.data)
SRE1_latt_dict[1] = calc_classical_magic(psi_magic.data)

magic_latt_dict[0] = {key: info_dict[0][key] - SRE1_latt_dict[0][key] for key in info_dict[0].keys()}
magic_latt_dict[1] = {key: info_dict[1][key] - SRE1_latt_dict[1][key] for key in info_dict[0].keys()}

nonint_magic[0] = non_integer_magic(info_dict[0])
nonint_magic[1] = non_integer_magic(info_dict[1])

total_info[0] = calc_total_info(magic_latt_dict[0])
total_info[1] = calc_total_info(magic_latt_dict[1])


print(nonint_magic)
print(total_info)
#%% Figure
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
for key1 in info_dict.keys():
    for key2 in info_dict[key1].keys():
        value = max(info_dict[key1][key2])
        if value > max_value:
            max_value = value
colormap = cm.ScalarMappable(norm=Normalize(vmin=0., vmax=max_value), cmap=color_map)

# Fig 1: Plot
fig1 = plt.figure(figsize=(20, 10))
fig1.suptitle('Info lattice', fontsize=20)
gs = GridSpec(1, 2, figure=fig1, hspace=0, wspace=0.1)

ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[0, 1])

plot_info_latt(info_dict[0], ax1, color_map, max_value=max_value, indicate_ints=True)
plot_info_latt(info_dict[1], ax2, color_map, max_value=max_value, indicate_ints=True)

# Fig 1: Colorbar
cbar_ax = fig1.add_subplot(gs[:, -1])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("left", size="5%", pad=0)
cbar = fig1.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()
cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)




# Fig 1: Plot
fig2 = plt.figure(figsize=(20, 10))
fig2.suptitle('SRE1 lattice', fontsize=20)
gs = GridSpec(1, 2, figure=fig2, hspace=0, wspace=0.1)

ax1_2 = fig2.add_subplot(gs[0, 0])
ax2_2 = fig2.add_subplot(gs[0, 1])

plot_magic_latt(SRE1_latt_dict[0], ax1_2, color_map, max_value=max_value, indicate_ints=True)
plot_magic_latt(SRE1_latt_dict[1], ax2_2, color_map, max_value=max_value, indicate_ints=True)

# Fig 1: Colorbar
cbar_ax = fig2.add_subplot(gs[:, -1])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("left", size="5%", pad=0)
cbar = fig2.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()
cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)



# Fig 1: Plot
fig3 = plt.figure(figsize=(20, 10))
fig3.suptitle('Magic lattice', fontsize=20)
gs = GridSpec(1, 2, figure=fig2, hspace=0, wspace=0.1)

ax1_3 = fig3.add_subplot(gs[0, 0])
ax2_3 = fig3.add_subplot(gs[0, 1])

plot_info_latt(magic_latt_dict[0], ax1_3, color_map, max_value=max_value, indicate_ints=True)
plot_info_latt(magic_latt_dict[1], ax2_3, color_map, max_value=max_value, indicate_ints=True)

# Fig 1: Colorbar
cbar_ax = fig3.add_subplot(gs[:, -1])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("left", size="5%", pad=0)
cbar = fig3.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()
cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)

plt.show()
