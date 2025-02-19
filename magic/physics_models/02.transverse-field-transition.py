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
from modules.transverse_field_Ising import Hamiltonian_transverse_field_Ising
from modules.InfoLattice import calc_info, plot_info_latt
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

theta = np.linspace(0.05, pi/2 - 0.05, 21)
L = 6

lmax_info = np.zeros((len(theta), ))
l0_avg_info = np.zeros((len(theta), ))
lmax_sre1 = np.zeros((len(theta), ))
l0_avg_sre1 = np.zeros((len(theta), ))
total_info = np.zeros((len(theta), ))
total_sre1 = np.zeros((len(theta), ))
single_qubit_sre1 = np.zeros((len(theta), ))
non_local_sre1 = np.zeros((len(theta), ))


disentanglers = minimal_clifford_disentanglers()
#%% Main

for i, angle in enumerate(theta):

    loger_main.info(f'theta: {i}/ {len(theta)- 1}')

    # Spectrum
    H = Hamiltonian_transverse_field_Ising(angle, L, boundary='Open')
    evals, evecs = np.linalg.eigh(H)
    gs_qiskit = Statevector(evecs[:, 0])

    # Info and SRE1 Lattice
    info_gs = calc_info(evecs[:, 0])
    sre1_gs = calc_SRE1_lattice(evecs[:, 0])

    # Minimised sre1
    psi_min, _ = minimise_entanglement(gs_qiskit, L, disentanglers)
    sre1_gs_min = calc_SRE1_lattice(psi_min.data)

    # Top and bottom info
    lmax_info[i]   = info_gs[L][0]
    l0_avg_info[i] = np.mean(info_gs[1])
    lmax_sre1[i]   = sre1_gs[L][0]
    l0_avg_sre1[i] = np.mean(sre1_gs[1])
    total_info[i] = calc_total_info(info_gs) / L
    total_sre1[i] = calc_total_info(sre1_gs) / L
    single_qubit_sre1[i] = np.mean(sre1_gs_min[1])

    for key in sre1_gs_min.keys():
        if key > 1:
            non_local_sre1[i] += np.sum(sre1_gs_min[key])
    non_local_sre1[i] = non_local_sre1[i] / L
#%% Figures

# Format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']

fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 1, figure=fig1, hspace=0.5)
ax0 = fig1.add_subplot(gs[0, 0])

ax0.plot(theta, lmax_info,   color=color_list[0], label='Top info')
ax0.plot(theta, l0_avg_info, color=color_list[1], label='Bottom info')
ax0.plot(theta, lmax_sre1,   color=color_list[2], label='Top sre1')
ax0.plot(theta, l0_avg_sre1, color=color_list[3], label='Bottom sre1')
ax0.plot(theta, total_info, color=color_list[4], marker='o', label='Total info')
ax0.plot(theta, total_sre1, color='k', marker='o', label='total sre1 info')
ax0.plot(theta, single_qubit_sre1,  color='m', label='single qubit sre1')
ax0.plot(theta, non_local_sre1, color='c', label='non local sre1')

ax0.set_xticks([0, pi/8, pi/4, 3 * pi/8,  np.pi/2], ['$0$', '$\\pi/8$', '$\\pi/4$', '$3\\pi/8$',  '$\\pi/2$'])
ax0.set_xlabel('$\\theta$')
ax0.set_ylabel('Information')
ax0.legend()
plt.show()