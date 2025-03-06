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
from modules.xyz_model import Hamiltonian_XYZ
from modules.InfoLattice_spin1 import calc_info, plot_info_latt
from modules.MagicLattice_spin1 import calc_SRE1_lattice, calc_total_info

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

D1 = np.linspace(-0.5, -0.3, 2)[:-1]
D2 = np.linspace(-0.3, 0.97, 10)[:-1]
D3 = np.linspace(0.97, 2, 5)
D = np.concatenate((D1, D2, D3))
L = 6
Jx = -1
Jy = -1
Jz = -1

lmax_info = np.zeros((len(D), ))
l0_avg_info = np.zeros((len(D), ))
lmax_sre1 = np.zeros((len(D), ))
l0_avg_sre1 = np.zeros((len(D), ))
total_info = np.zeros((len(D), ))
total_sre1 = np.zeros((len(D), ))

#%% Main

for i, d in enumerate(D):

    loger_main.info(f'D: {i}/ {len(D)- 1}')

    # Spectrum
    H = Hamiltonian_XYZ(L, Jx, Jy, Jz, d)
    evals, evecs = np.linalg.eigh(H)
    gs_qiskit = Statevector(evecs[:, 0])

    # Info and SRE1 Lattice
    info_gs = calc_info(evecs[:, 0])
    sre1_gs = calc_SRE1_lattice(evecs[:, 0])

    # Top and bottom info
    lmax_info[i]   = info_gs[L][0] + info_gs[L - 1][0] + info_gs[L - 1][1]
    l0_avg_info[i] = np.mean(info_gs[1])
    lmax_sre1[i]   = sre1_gs[L][0] + sre1_gs[L - 1][0] + sre1_gs[L - 1][1]
    l0_avg_sre1[i] = np.mean(sre1_gs[1])
    total_info[i] = calc_total_info(info_gs) / L
    total_sre1[i] = calc_total_info(sre1_gs) / L

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

ax0.plot(D, lmax_info,   color=color_list[0], label='Top info')
ax0.plot(D, l0_avg_info, color=color_list[1], label='Bottom info')
ax0.plot(D, lmax_sre1,   color=color_list[2], label='Top sre1')
ax0.plot(D, l0_avg_sre1, color=color_list[3], label='Bottom sre1')
ax0.plot(D, total_info, color=color_list[4], marker='o', label='Total info')
ax0.plot(D, total_sre1, color='k', marker='o', label='total sre1 info')

ax0.set_xlabel('$D$')
ax0.set_ylabel('Information')
ax0.legend()
plt.show()