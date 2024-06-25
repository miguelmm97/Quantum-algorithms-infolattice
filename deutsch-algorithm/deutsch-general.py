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

# Information lattice
from functions import (deutsch_function, calc_info, plot_info_latt, calc_info_per_scale, get_fileID, store_my_data,
                       attr_my_data, median)

#%% Definitions and parameters

# Circuit
input_state = '000'
n_qubits = len(input_state) - 1
black_box, type_func = deutsch_function(n_qubits, return_info=True)

# Information
n_steps = 3
neg_tol = -1e-12
info_dict = {}

# Circuit instance
qc = QuantumCircuit(n_qubits + 1)
psi0 = Statevector.from_label(input_state)

#%% Debug
str_state = '10'
state_aux = Statevector.from_label(str_state)
qc2 = QuantumCircuit(len(str_state))

def add_cx(qc0, bit_string):
    for qubit, bit in enumerate(reversed(bit_string)):
        if bit == "1": qc0.x(qubit)
    return qc0


qc2.h(range(len(str_state)))
qc2.barrier()
qc2 = add_cx(qc2, f'{1:0b}')
qc2.mcx(list(range(len(str_state) - 1)), len(str_state)-1)
qc2 = add_cx(qc2, f'{1:0b}')

final_state = state_aux.evolve(qc2)
qc2.draw(output='mpl')
plt.show()


#%% Algorithm evolution and information

# Step 1: Hadamard transform in the control and target qubits
qc.x(n_qubits)
qc.h(range(n_qubits + 1))
info_dict[0] = calc_info(psi0.evolve(qc).data)
state = psi0.evolve(qc)

# Step 2: Application of the Deutsch function
qc.compose(black_box, inplace=True)
info_dict[1] = calc_info(psi0.evolve(qc).data)
state = psi0.evolve(qc)

# Step 3: Invert hadamard transform on the target
qc.h(range(n_qubits))
info_dict[2] = calc_info(psi0.evolve(qc).data)
state = psi0.evolve(qc)

# Debug
for i in range(n_steps):
    for key in info_dict[i].keys():
        if any(info_dict[i][key] < neg_tol):
            raise ValueError(f'Negative mutual information: step {i} | scale {key}')

#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# palette1 = seaborn.color_palette(palette='Blues', n_colors=n_iter+1)
# palette2 = seaborn.color_palette(palette='dark:salmon_r', n_colors=n_iter+1)

# Circuit
fig1 = qc.draw(output="mpl")
ax1 = fig1.gca()
ax1.set_title(f'Deutsch Algorithm with a {type_func} function')


# Information lattice per iteration
fig2 = plt.figure(figsize=(8, 5))
gs = GridSpec(1, n_steps, figure=fig2)
for i in range(n_steps):
    ax2 = fig2.add_subplot(gs[0, i])
    plot_info_latt(info_dict[i], ax2)
    ax2.set_title(f't: {i / (n_steps - 1):.2f}')

fig2.suptitle(f'Information lattice in Deutchs algorithm for a {type_func} function')
plt.show()