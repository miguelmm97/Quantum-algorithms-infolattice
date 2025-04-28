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
from qiskit.quantum_info import Statevector, DensityMatrix

# Information lattice and functions
from modules.InfoLattice import calc_info, plot_info_latt
from modules.StabilizerLattice import calc_stabilizer_vN_info, plot_stabilizer_vN_latt, stabilizer_vN_entropy, partial_trace
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

#%% Main

# Parameters
num_qubits = 3
psi0 = Statevector.from_label('0' * num_qubits)

# Circuit
circuit = QuantumCircuit(num_qubits)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)
circuit.ry(np.pi/4, 0)
circuit.ry(np.pi/4, 2)

# Information lattices
vN_lattice = calc_info(psi0.evolve(circuit).data)
stab_vN_lattice = calc_stabilizer_vN_info(psi0.evolve(circuit).data)
magic_lattice = {key: vN_lattice[key] - stab_vN_lattice[key] for key in vN_lattice.keys()}

# Debug
# psi = psi0.evolve(circuit).data
# rho = np.outer(psi, psi.conj())
# rho_01 = partial_trace(rho, 0, 1)
# rho_11 = partial_trace(rho, 1, 1)
# rho_21 = partial_trace(rho, 2, 1)
# rho_02 = partial_trace(rho, 0, 2)
# rho_12 = partial_trace(rho, 1, 2)
# rho_03 = partial_trace(rho, 0, 3)
# S_01, entropies_01, list_strings_01 = stabilizer_vN_entropy(rho_01, return_entropy_list=True)
# S_11, entropies_11, list_strings_11 = stabilizer_vN_entropy(rho_11, return_entropy_list=True)
# S_21, entropies_21, list_strings_21 = stabilizer_vN_entropy(rho_21, return_entropy_list=True)
# S_02, entropies_02, list_strings_02 = stabilizer_vN_entropy(rho_02, return_entropy_list=True)
# S_12, entropies_12, list_strings_12 = stabilizer_vN_entropy(rho_12, return_entropy_list=True)
# S_03, entropies_03, list_strings_03 = stabilizer_vN_entropy(rho, return_entropy_list=True)
# min_index = np.where(entropies_03 == np.min(entropies_03))[0][0]
# min_group = list_strings_03[min_index]

#%% Figures

# Format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']

# Defining a colormap without negatives and white color for no info
color_map = plt.get_cmap("PuOr").reversed()
colors = color_map(np.linspace(0, 1, 41)[20:])
colors[0] = [1, 1, 1, 1]
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)
colormap = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=2), cmap=color_map)

fig1 = plt.figure()
gs = GridSpec(1, 3, figure=fig1, hspace=0.5)
ax0 = fig1.add_subplot(gs[0, 0])
ax1 = fig1.add_subplot(gs[0, 1])
ax2 = fig1.add_subplot(gs[0, 2])

plot_info_latt(vN_lattice, ax0, color_map, indicate_ints=True)
plot_info_latt(stab_vN_lattice, ax1, color_map, indicate_ints=True)
plot_info_latt(magic_lattice, ax2, color_map, indicate_ints=True)
ax0.set_title(f'Information Lattice')
ax1.set_title(f'Stabilizer Information Lattice')
ax2.set_title(f'Magic Information Lattice')

# Fig 1: Colorbar
cbar_ax = fig1.add_subplot(gs[-1, :])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("bottom", size="5%", pad=0)
cbar = fig1.colorbar(colormap, cax=cax, orientation='horizontal')
cbar_ax.set_axis_off()
cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)
plt.show()

