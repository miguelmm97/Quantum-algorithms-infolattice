#%% Imports

# Managing data
import os
import sys
import h5py
from datetime import date

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix

import numpy as np
from numpy import pi
from itertools import permutations, product
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Information lattice and functions
from modules.InfoLattice import S_vN
from modules.MagicLattice import calc_SRE1_pure, measurement_outcome_allstab_shannon
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
theta = [0] #np.linspace(0, pi, 101)
phi = [0] # np.linspace(0, 2 * pi, 11)
index = [p for p in product(np.arange(len(theta)), np.arange(len(phi)))]
SRE1_1q = np.zeros((len(index), ))
SVN = np.zeros((len(index), ))
shannon_measurement_1q = np.zeros((len(index), 4 - 1))
psi0_1 = Statevector.from_label('0')

for i, indices in enumerate(index):
    loger_main.info(f'Calculating state {i}/ {len(index) - 1}')
    circuit_1q = QuantumCircuit(1)
    circuit_1q.ry(theta[indices[0]], 0)
    SRE1_1q[i] = calc_SRE1_pure(psi0_1.evolve(circuit_1q).data, remove_I=False)
    # SVN[i]  = S_vN(psi0_1.evolve(circuit_1q).data)
    evolved_density = DensityMatrix(psi0_1.evolve(circuit_1q)).data
    shannon_measurement_1q[i, :], list_strings_1q = measurement_outcome_allstab_shannon(evolved_density)


# Two-qubit case
theta1 = [0] # np.linspace(0, pi, 101)
theta2 = 0
SRE1_2q = np.zeros((len(index), ))
circuit0 = QuantumCircuit(2)
# circuit0.ry(theta2, 1)
# circuit0.h(0)
# circuit0.cx(0, 1)
psi0_2q = Statevector.from_label('00').evolve(circuit0)

for i, indices in enumerate(index):
    loger_main.info(f'Calculating state {i}/ {len(index) - 1}')
    circuit_2q = QuantumCircuit(2)
    # circuit_2q.rx(theta[indices[0]], 1)
    # circuit_2q.ry(theta[indices[0]], 0)
    SRE1_2q[i] = calc_SRE1_pure(psi0_2q.evolve(circuit_2q).data, remove_I=False)
    # SVN[i]  = S_vN(psi0_1.evolve(circuit_1q).data)
    evolved_density = DensityMatrix(psi0_2q.evolve(circuit_2q)).data
    if i==0:
        shannon_entropies = measurement_outcome_allstab_shannon(evolved_density)[0]
        num_entropies = len(shannon_entropies)
        shannon_measurement_2q = np.zeros((len(index), num_entropies))
    shannon_measurement_2q[i, :], list_strings_2q = measurement_outcome_allstab_shannon(evolved_density)


# circuit_1q = QuantumCircuit(1)
# circuit_1q.ry(pi/4, 0)
# psi = psi0_1.evolve(circuit_1q).data
# SRE1 = calc_SRE1_pure(psi)
# shannon_measurement, strings = measurement_outcome_shannon(psi)


#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']
marker_list = np.tile(['s', 'd', 'o', '*', '^', 'x', '<'], 10)
fontsize=15


fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 1, figure=fig1, hspace=0.5)
ax0 = fig1.add_subplot(gs[0, 0])

# ax0.plot(theta, 1 - SRE1_1q, color='Blue', marker='s', label='$SRE1$')
# for i in range(len(shannon_measurement_1q[0, :])):
#     ax0.plot(theta, 1 - shannon_measurement_1q[:, i], marker='', linestyle='dashed', label=list_strings_1q[i])
# ax0.set_xticks([0, pi/4,  pi/2, 3 * pi /4, pi], ['$0$', '$\\pi/4$', '$\\pi/2$', '$3\\pi/4$',  '$\\pi$'])
# ax0.set_title('1 qubit comparison')
# ax0.legend()

list_labels = []
for i in list_strings_2q:
    list_labels.append(i[0] + '-' + i[1])


fig2 = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 1, figure=fig2, hspace=0.5)
ax0 = fig2.add_subplot(gs[0, 0])
#
# pauli_strings = []
# pauli_iter = product('XYZ', repeat=2)
# for element in pauli_iter:
#     pauli_strings.append(''.join(element))
print(len(list_strings_2q))
# ax0.plot(theta, 2 - SRE1_2q, marker='s', label='$SRE1$', markerfacecolor='none', markeredgecolor='Blue', linestyle='None')
ax0.plot(np.linspace(0, len(list_strings_2q), len(list_strings_2q)), 2 - shannon_measurement_2q[0, :], marker='o', linestyle='None')
ax0.set_xticks(np.linspace(0, len(list_labels), len(list_strings_2q)), list_labels, rotation='vertical')
ax0.grid(axis='x')
ax0.set_title('Bell pair: 00 + 11')
# ax0.legend()
plt.show()
