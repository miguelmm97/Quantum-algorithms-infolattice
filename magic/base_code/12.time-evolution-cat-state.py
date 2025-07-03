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
from modules.InfoLattice import calc_info, plot_info_latt, calc_entropies
from modules.functions import *


#%% Main

num_qubits = 14
psi00 = Statevector.from_label('00' * int(num_qubits * 0.5))
psi01 = Statevector.from_label('01' * int(num_qubits * 0.5))
psi10 = Statevector.from_label('10' * int(num_qubits * 0.5))
psi11 = Statevector.from_label('11' * int(num_qubits * 0.5))

t = 0.023
alpha00 = np.sqrt(1/10)
alpha01 = np.sqrt((3 ** (-1/4)) / 10)
alpha10 = np.sqrt(t)
alpha11 = np.sqrt(1 - alpha00 ** 2 - alpha01 ** 2 - alpha10 ** 2)
#alpha00 = 1/2
#alpha01 = 1/2
#alpha10 = 1/2
#alpha11 = 1/2
psi = alpha00 * psi00 + alpha01 * psi01 + alpha10 * psi10 + alpha11 * psi11
# num_coeffs_cat = np.where(np.abs(psi) > 0.)[0]

# qc = QuantumCircuit(num_qubits)
# qc.ry(np.pi/4, 6)
# qc.rx(np.pi/5, 5)
# qc.cx(6, 5)
# psi = psi.evolve(qc)
# num_coeffs = np.where(np.abs(psi) > 0.)[0]


print(alpha00 **2 + alpha01 ** 2 + alpha10 ** 2 + alpha11 ** 2)
info_lattice = calc_info(psi.data)
entropy_lattice = calc_entropies(psi.data)
total_info = np.sum([np.sum(info_lattice[i]) for i in info_lattice.keys()])

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
plt.show()

