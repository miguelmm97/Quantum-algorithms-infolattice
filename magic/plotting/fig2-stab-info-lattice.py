#%% Imports

# Built-in modules
import math
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import FancyArrowPatch

# Managing data
import os
import sys
import h5py
from datetime import date

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace

# Information lattice and functions
from modules.InfoLattice import calc_info, plot_info_latt, calc_info_per_scale
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
n_qubits = 16
psi0_label = '0' * n_qubits
psi0 = Statevector.from_label(psi0_label)

# Circuit parameters
Nlayers = 30
qubits = list(range(n_qubits))

#%% Circuits
clifford = random_clifford_circuit(num_qubits=n_qubits, depth=Nlayers)
psi_stab = psi0.evolve(clifford)
info_latt_stab = calc_info(psi_stab.data)
info_per_scale_stab = calc_info_per_scale(info_latt_stab, bc='open')

GHZ_circuit = QuantumCircuit(n_qubits)
GHZ_circuit.h(0)
for i in range(1, n_qubits):
    GHZ_circuit.cx(i - 1, i)
GHZ = psi0.evolve(GHZ_circuit)
info_latt_GHZ = calc_info(GHZ.data)
info_per_scale_GHZ = calc_info_per_scale(info_latt_GHZ, bc='open')

scales = np.arange(0, n_qubits)
interp_func = PchipInterpolator(scales, info_per_scale_stab / np.arange(n_qubits + 1, 1, -1))
interp_scales_stab = np.arange(scales[0], scales[-1] + 0.1, 0.1)
interp_average_stab = interp_func(interp_scales_stab)

interp_func = PchipInterpolator(scales, info_per_scale_GHZ / np.arange(n_qubits + 1, 1, -1))
interp_scales_GHZ = np.arange(scales[0], scales[-1] + 0.1, 0.1)
interp_average_GHZ= interp_func(interp_scales_GHZ)



#%% Figures
# Format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']
fontsize=20

#
colors = ['white', 'deepskyblue', 'mediumpurple']
colormap = LinearSegmentedColormap.from_list('stab_colormap', colors, N=3)
colormap_cbar = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=2), cmap=colormap)


fig1 = plt.figure(figsize=(15, 5))
gs = GridSpec(1, 4, figure=fig1, wspace=0.05)
ax0 = fig1.add_subplot(gs[0, 0])
ax1 = fig1.add_subplot(gs[0, 1])
ax2 = fig1.add_subplot(gs[0, 2])
ax0_inset = ax0.inset_axes([0, 1.1, 1, 0.2], )
# ax1_inset = ax1.inset_axes([0.95, 0.45, 0.5, 0.3], )

# Plots
plot_info_latt(info_latt_stab, ax0, colormap, indicate_ints=False)
ax0.set_xlabel('$n$', fontsize=fontsize, labelpad=-15)
ax0.set_ylabel('$\ell$', fontsize=fontsize, labelpad=-15)
ax0.axis('on')
ax0.set_xticks(ticks=[0.025, 0.975], labels=['0', f'{n_qubits}'])
ax0.set_yticks(ticks=[0.025, 0.975], labels=['0', f'{n_qubits - 1}'])
ax0.tick_params(which='major', width=0.75, labelsize=fontsize)

plot_info_latt(info_latt_GHZ, ax1, colormap, indicate_ints=False)
ax1.set_xlabel('$n$', fontsize=fontsize, labelpad=-15)
# ax1.set_ylabel('$\ell$', fontsize=fontsize, labelpad=-15)
ax1.axis('on')
ax1.set_xticks(ticks=[0.025, 0.975], labels=['0', f'{n_qubits}'])
ax1.set_yticks(ticks=[0.025, 0.975], labels=['', ''])
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)




ax0_inset.plot(interp_scales_stab, interp_average_stab, marker='None', color='royalblue')
# ax1_inset.plot(interp_scales_GHZ, interp_average_GHZ, marker='None', color='royalblue')
ax0_inset.fill_between(interp_scales_stab, interp_average_stab, color='royalblue', alpha=0.2)
# ax1_inset.fill_between(interp_scales_GHZ, interp_average_GHZ, color='royalblue', alpha=0.2)
ax0_inset.plot(0.5 * n_qubits * np.ones((10, )) - 1, np.linspace(0, 1, 10), '--', color='black', alpha=0.25)
# ax1_inset.plot(0.5 * n_qubits * np.ones((10, )) - 1, np.linspace(0, 1, 10), '--', color='black', alpha=0.25)


# Style
ax0_inset.set_xlim(0, n_qubits - 1)
ax0_inset.set_ylim(0, 1)
ax0_inset.set_xlabel('$\ell$', fontsize=fontsize-5, labelpad=-10)
ax0_inset.set_ylabel('$\langle I^\ell \\rangle$', fontsize=fontsize-5, rotation='horizontal')
label = ax0_inset.yaxis.get_label()
x, y = label.get_position()
label.set_position((x, y - 0.2))
label = ax0_inset.xaxis.get_label()
x, y = label.get_position()
label.set_position((x + 0.2, y + 1))
ax0_inset.set_xticks(ticks=[0, (n_qubits / 2) - 1, n_qubits - 1], labels=['$0$', '$L/2$', '$L$'])
ax0_inset.set_yticks(ticks=[0, 1])
yminor_ticks = [0.5]
ax0_inset.yaxis.set_minor_locator(plt.FixedLocator(yminor_ticks))
ax0_inset.tick_params(which='major', width=0.75, labelsize=fontsize-5)
ax0_inset.tick_params(which='major', length=6, labelsize=fontsize-5)
ax0_inset.tick_params(which='minor', width=0.75, labelsize=fontsize-5)
ax0_inset.tick_params(which='minor', length=3, labelsize=fontsize-5)
#
# ax1_inset.set_xlim(0, n_qubits - 1)
# ax1_inset.set_ylim(0, 1)
# ax1_inset.set_xlabel('$\ell$', fontsize=fontsize, labelpad=-10)
# ax1_inset.set_ylabel('$\langle I^\ell \\rangle$', fontsize=fontsize, rotation='horizontal')
# label = ax1_inset.yaxis.get_label()
# x, y = label.get_position()
# label.set_position((x, y - 0.2))
# label = ax1_inset.xaxis.get_label()
# x, y = label.get_position()
# label.set_position((x + 0.2, y + 1))
# ax1_inset.set_xticks(ticks=[0, (n_qubits / 2) - 1, n_qubits - 1], labels=['$0$', '$L/2$', '$L$'])
# ax1_inset.set_yticks(ticks=[0, 1])
# yminor_ticks = [0.5]
# ax1_inset.yaxis.set_minor_locator(plt.FixedLocator(yminor_ticks))
# ax1_inset.tick_params(which='major', width=0.75, labelsize=fontsize)
# ax1_inset.tick_params(which='major', length=6, labelsize=fontsize)
# ax1_inset.tick_params(which='minor', width=0.75, labelsize=fontsize)
# ax1_inset.tick_params(which='minor', length=3, labelsize=fontsize)
# ax0.text(0.3, 1, '$(a)$', fontsize=fontsize)
# ax1.text(0.3, 1, '$(b)$', fontsize=fontsize)
# ax0_inset.text(8, 0.7, '$\Gamma=0$', fontsize=fontsize)
# ax1_inset.text(8, 0.7, '$\Gamma=1$', fontsize=fontsize)

# Legend
# ax_legend1 = fig1.add_axes([0.47, 0.7, 0.05, 0.05])
# ax_legend1.scatter([0], [0], color=colors[0], s=100, marker='o', edgecolor='black')
# ax_legend1.set_xticks([])
# ax_legend1.set_yticks([])
# ax_legend1.set_axis_off()
# ax_legend1.text(0.05, -0.05, '$0$', fontsize=fontsize)
# ax_legend1.text(0.03, 0.2, '$\\underline{i_n^\ell}$', fontsize=fontsize)
#
# ax_legend2 = fig1.add_axes([0.47, 0.6, 0.05, 0.05])
# ax_legend2.scatter([0], [0], color=colors[1], s=100, marker='o', edgecolor='black')
# ax_legend2.set_xticks([])
# ax_legend2.set_yticks([])
# ax_legend2.set_axis_off()
# ax_legend2.text(0.05, -0.05, '$1$', fontsize=fontsize)
#
#
# ax_legend3 = fig1.add_axes([0.47, 0.5, 0.05, 0.05])
# ax_legend3.scatter([0], [0], color=colors[2], s=100, marker='o', edgecolor='black')
# ax_legend3.set_xticks([])
# ax_legend3.set_yticks([])
# ax_legend3.set_axis_off()
# ax_legend3.text(0.05, -0.05, '$2$', fontsize=fontsize)

fig1.savefig('fig-stab.pdf', format='pdf')
plt.show()

