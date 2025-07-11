#%% Imports

# Built-in modules
import math
import numpy as np
from scipy.sparse.linalg import eigsh
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
from modules.InfoLattice_spin1 import calc_info, plot_info_latt, calc_entropies
from modules.functions import *
from modules.potts_model import Hamiltonian_potts

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
g = 10
J = 1
L = 5

H = Hamiltonian_potts(L, g, J)
eigvals, eigstates = np.linalg.eigh(H)
ground_state = eigstates[:, 0]
# energy, ground_state = eigsh(H, k=1, which='SA')
# ground_state = ground_state[:, 0]

entropy_lattice = calc_entropies(ground_state)
info_lattice = calc_info(ground_state)
total_info = np.sum([np.sum(info_lattice[i]) for i in info_lattice.keys()])

loger_main.info(f'Hamiltonian is Hermitian: {np.allclose(H, H.T.conj())}')
loger_main.info(f'Ground state is normalised: {np.allclose(1, np.sqrt(np.sum(ground_state * ground_state.T.conj())))}')
loger_main.info(f'Total info is the number of qutrits: {np.allclose(total_info, L)}')




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
gs = GridSpec(1, 1, figure=fig1, hspace=0.5)
ax0 = fig1.add_subplot(gs[0, 0])
plot_info_latt(info_lattice, ax0, color_map, indicate_ints=True, tol_ints=1e-2)
ax0.set_title(f'Information Lattice. Total info: {total_info}')

# Fig 1: Colorbar
cbar_ax = fig1.add_subplot(gs[-1, :])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("bottom", size="5%", pad=0)
cbar = fig1.colorbar(colormap, cax=cax, orientation='horizontal')
cbar_ax.set_axis_off()
cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)


fig2 = plt.figure()
gs = GridSpec(1, 1, figure=fig2, hspace=0.5)
ax0 = fig2.add_subplot(gs[0, 0])
plot_info_latt(entropy_lattice, ax0, color_map, indicate_ints=True, tol_ints=1e-2)
ax0.set_title(f'Information Lattice. Total info: {total_info}')

# Fig 1: Colorbar
cbar_ax = fig2.add_subplot(gs[-1, :])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("bottom", size="5%", pad=0)
cbar = fig2.colorbar(colormap, cax=cax, orientation='horizontal')
cbar_ax.set_axis_off()
cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)



fig3 = plt.figure()
gs = GridSpec(1, 1, figure=fig3, hspace=0.5)
ax0 = fig3.add_subplot(gs[0, 0])
ax0.plot(np.arange(len(eigvals)), eigvals, marker='o', linestyle='none')




plt.show()


