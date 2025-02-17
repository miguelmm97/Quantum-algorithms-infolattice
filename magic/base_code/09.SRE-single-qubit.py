#%% Imports

# Managing data
import os
import sys
import h5py
from datetime import date

# Imports from Qiskit
from qiskit import QuantumCircuit

import numpy as np
from numpy import pi

# Information lattice and functions
from modules.InfoLattice import plot_info_latt, calc_info
from modules.MagicLattice import shannon
from modules.functions import *

from matplotlib.colors import LinearSegmentedColormap

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

# Single qubit magic
theta = np.linspace(0, pi, 100)
phi = np.linspace(0, 2 * pi, 100)
SRE1 = np.zeros((len(theta), len(phi)))
psi0_1 = Statevector.from_label('0')

for i, angle_y in enumerate(phi):
    for j, angle_z in enumerate(theta):
        loger_main.info(f'Calculating angles phi: {i} / {len(phi) - 1}, theta: {j} / {len(theta) - 1}')
        circuit_1q = QuantumCircuit(1)
        circuit_1q.ry(angle_y, 0)
        circuit_1q.rz(angle_z, 0)
        SRE1[i, j] = shannon(psi0_1.evolve(circuit_1q).data)

max_idx = np.where(SRE1 == np.max(SRE1))
max_theta = theta[max_idx[1][0]]
max_phi = phi[max_idx[0][0]]

# Two qubit addition of SRE1
psi0_2 = Statevector.from_label('00')
circuit_1q = QuantumCircuit(1)
circuit_1q.ry(max_phi- 0.3 * pi, 0)
circuit_1q.rz(max_theta - 0.2 * pi, 0)
SRE1_max1q = shannon(psi0_1.evolve(circuit_1q).data)
circuit_2q = QuantumCircuit(2)
circuit_2q.ry(max_phi, [0, 1])
circuit_2q.rz(max_theta, [0, 1])
SRE1_max2q = shannon(psi0_2.evolve(circuit_2q).data)
loger_main.info(f'Maximum value of SRE1 for two qubits: {SRE1_max2q :.2f}')
loger_main.info(f'Two times maximum value of SRE1 for one qubit: {2 * SRE1_max1q :.2f}')




#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']
fontsize=15

X, Y = np.meshgrid(theta, phi)

# Plot the heatmap
fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 1, figure=fig1, hspace=0.5)
ax0 = fig1.add_subplot(gs[0, 0])

ax0.pcolormesh(X, Y, SRE1, shading='auto', cmap='viridis')
ax0.scatter(max_theta, max_phi, facecolor='red')
ax0.set_xlabel(f'$\\theta$', fontsize=fontsize)
ax0.set_ylabel(f'$\\phi$', fontsize=fontsize)
ax0.set_title(f'SRE1 max: {np.max(SRE1) :.2f}', fontsize=fontsize)
ax0.set_xticks([0, np.pi/2, np.pi], ["$0$", "$\\pi/2$", "$\\pi$"], fontsize=fontsize)
ax0.set_yticks([0, np.pi, 2*np.pi], ["$0$", "$\\pi$", "$2\\pi$"], fontsize=fontsize)
fig1.gca().set_aspect(0.5)

# Defining a colormap
max_value = np.max(SRE1)
color_map = plt.get_cmap("viridis")
colors = color_map(np.linspace(0, 1, 41))
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)
colormap = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=max_value), cmap=color_map)
cbar_ax = fig1.add_subplot(gs[0, -1])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("right", size="5%", pad=1)
cbar = fig1.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()
cbar.set_label(label='$SRE1$', labelpad=10, fontsize=20)
plt.show()




