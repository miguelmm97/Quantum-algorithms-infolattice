#%% Imports

# Math and plotting
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn

# Managing data
import h5py
import os
import sys
from datetime import date

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, HGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator

# Algorithm
from functions import qft_circuit, Umod_multi,  calc_info, plot_info_latt, calc_info_per_scale

#%% Variables
# Variables
m = int(np.ceil(np.log2(15)))   # Qubits in the second register (phase estimation)
t = 3                           # Qubits in the first register (QFT)
n_iter = t + 1                  # Number of information measurements
register1_0 = '0' * t           # First register initial state
register2_0 = '1' * m           # Second register initial state
state = Statevector.from_label(register1_0 + register2_0) # Initial state
info_dict = {}
title_dict = {}

#%% Main: U(x) = 7x mod 15

# Create 7mod15 gate
Umod = QuantumCircuit(m)
Umod.x(range(m))
Umod.swap(1, 2)
Umod.swap(2, 3)
Umod.swap(0, 3)
Umod = Umod.to_gate()

# Repeated application of the powers of the U operator
qc_shor = QuantumCircuit(t + m, t)
qc_shor.h(range(t))
qc_shor.x(t)
state = state.evolve(qc_shor)
info_dict[0] = calc_info(state.data)
title_dict[0] = 'H + X'
qc_shor.barrier(label='H + X')

for idx in range(t - 1):
    # Information lattice
    state = state.evolve(Umod_multi(m, idx, Umod), [idx] + list(range(t, t + m)))
    info_dict[idx + 1] = calc_info(state.data)
    title_dict[idx + 1] = f'U({2 ** idx})'
    # Plotting the circuit
    qc_shor.append(Umod_multi(m, idx, Umod), [idx] + list(range(t, t + m)))
    qc_shor.barrier(label=f'U({2 ** idx})')

# Inverse Quantum Fourier transform
qft_inv = qft_circuit(t).inverse()
qft_inv.name = 'QFT$^\dagger$'
state = state.evolve(qft_inv)
info_dict[n_iter - 1] = calc_info(state.data)
title_dict[n_iter - 1] = 'QFT$^\dagger$'
qc_shor.append(qft_inv, range(t))
qc_shor.barrier(label='QFT$^\dagger$')


#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


fig1 = qc_shor.draw(output="mpl", style="iqp")
ax1 = fig1.gca()

fig2 = plt.figure(figsize=(8, 5))
gs = GridSpec(2, int(np.ceil(n_iter / 2) + 1), figure=fig2)
for i in range(n_iter):
    if ((n_iter + 1) % 2) == 0:
        if i < int((n_iter + 1)/ 2):
            ax = fig2.add_subplot(gs[0, i])
        else:
            ax = fig2.add_subplot(gs[1, i % int((n_iter + 1) / 2)])
    else:
        if i <= int((n_iter + 1) / 2):
            ax = fig2.add_subplot(gs[0, i])
        elif i != (n_iter + 1) - 1:
            ax = fig2.add_subplot(gs[1, (i % int((n_iter + 1) / 2)) - 1])
        else:
            ax = fig2.add_subplot(gs[1, int((n_iter + 1) / 2) - 1])
    plot_info_latt(info_dict[i], ax)
    ax.set_title(title_dict[i])
plt.show()




