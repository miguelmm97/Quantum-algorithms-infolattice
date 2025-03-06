#%% Imports

# Managing data
import os
import sys
import h5py
from datetime import date

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import numpy as np
from numpy import pi
from itertools import permutations, product
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Information lattice and functions
from modules.InfoLattice import S_vN
from modules.MagicLattice import calc_SRE1_pure, measurement_outcome_shannon
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

# Single qubit case
theta = np.linspace(0, pi, 101)
phi = [0] # np.linspace(0, 2 * pi, 11)
index = [p for p in product(np.arange(len(theta)), np.arange(len(phi)))]

SRE1 = np.zeros((len(index), ))
SVN = np.zeros((len(index), ))
shannon_measurement = np.zeros((len(index), 4 - 1))
psi0_1 = Statevector.from_label('0')
#
for i, indices in enumerate(index):
    loger_main.info(f'Calculating state {i}/ {len(index) - 1}')
    circuit_1q = QuantumCircuit(1)
    circuit_1q.ry(theta[indices[0]], 0)
    circuit_1q.rz(phi[indices[1]], 0)
    SRE1[i] = calc_SRE1_pure(psi0_1.evolve(circuit_1q).data)
    # SVN[i]  = S_vN(psi0_1.evolve(circuit_1q).data)
    shannon_measurement[i, :] = measurement_outcome_shannon(psi0_1.evolve(circuit_1q).data)[0]

# circuit_1q = QuantumCircuit(1)
# circuit_1q.rx(pi/4, 0)
# psi = psi0_1.evolve(circuit_1q).data
# SRE1 = calc_SRE1_pure(psi)
# shannon_measurement, strings = measurement_outcome_shannon(psi)



#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']
fontsize=15


# Plot the heatmap
fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 1, figure=fig1, hspace=0.5)
ax0 = fig1.add_subplot(gs[0, 0])

# ax0.plot(np.arange(len(index)), SVN, color='Blue', marker='o', label='$S_{VN}$')
ax0.plot(np.arange(len(index)), 1 - SRE1, color='Blue', marker='s', label='$SRE1$')
for i in range(len(shannon_measurement[0, :])):
    ax0.plot(np.arange(len(index)), 1 - shannon_measurement[:, i], marker='', linestyle='dashed')

plt.show()
