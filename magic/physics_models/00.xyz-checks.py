#%% Imports

# Managing data
import os
import sys
import h5py
from datetime import date

# Imports from Qiskit
from qiskit.quantum_info import Statevector

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Information lattice and functions
from modules.functions import *
from modules.xyz_model import Hamiltonian_XYZ, spin
from modules.InfoLattice_spin1 import calc_info, plot_info_latt
from modules.MagicLattice import calc_SRE1_lattice, calc_total_info, minimal_clifford_disentanglers, minimise_entanglement

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

Jx = -1
Jy = -1
Jz = +20
D = 0.1
L = 6

#%% Main


H = Hamiltonian_XYZ(L, Jx, Jy, Jz, D)
evals, evecs = np.linalg.eigh(H)
print(evals[0], evals[1])

# gs_qiskit = Statevector(evecs[:, 0])
info_gs = calc_info(evecs[:, 0])
info_gs2 = calc_info(evecs[:, 1])
# sre1_gs = calc_SRE1_lattice(evecs[:, 0])
# sre1_gs2 = calc_SRE1_lattice(evecs[:, 1])
#
# disentanglers = minimal_clifford_disentanglers()
# psi_min, _ = minimise_entanglement(gs_qiskit, L, disentanglers)
# SRE1_infolatt_min = calc_SRE1_lattice(psi_min.data)
# VN_infolatt_min = calc_info(psi_min.data)



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
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)      # Colormap for the info lattice
colormap = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=2), cmap=color_map)  # Colormap for the colorbar


fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 2, figure=fig1, hspace=0.5)
ax0 = fig1.add_subplot(gs[0, 0])
ax1 = fig1.add_subplot(gs[0, 1])
fig1.suptitle('Info')
plot_info_latt(info_gs, ax0, color_map, indicate_ints=True, max_value=2)
plot_info_latt(info_gs2, ax1, color_map, indicate_ints=True, max_value=2)
# plot_info_latt(VN_infolatt_min, ax1, color_map, indicate_ints=True, max_value=2)

cbar_ax = fig1.add_subplot(gs[-1, :])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("bottom", size="5%", pad=0)
cbar = fig1.colorbar(colormap, cax=cax, orientation='horizontal')
cbar_ax.set_axis_off()
cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)


fig2 = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 1, figure=fig2, hspace=0.5)
ax0 = fig2.add_subplot(gs[0, 0])
ax0.plot(np.arange(len(evals)), evals, 'ob')


#
# #
# fig3 = plt.figure(figsize=(8, 6))
# gs = GridSpec(1, 2, figure=fig3, hspace=0.5)
# ax0 = fig3.add_subplot(gs[0, 0])
# ax1 = fig3.add_subplot(gs[0, 1])
# plot_info_latt(sre1_gs, ax0, color_map, indicate_ints=True , max_value=2)
# plot_info_latt(SRE1_infolatt_min, ax1, color_map, indicate_ints=True, max_value=2)
# fig3.suptitle('SRE1')
#
# cbar_ax = fig3.add_subplot(gs[-1, :])
# divider = make_axes_locatable(cbar_ax)
# cax = divider.append_axes("bottom", size="5%", pad=0)
# cbar = fig3.colorbar(colormap, cax=cax, orientation='horizontal')
# cbar_ax.set_axis_off()
# cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)
plt.show()

