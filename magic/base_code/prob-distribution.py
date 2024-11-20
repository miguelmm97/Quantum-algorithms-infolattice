#%% Imports

# Built-in modules
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Information lattice and functions
from modules.MagicLattice import calc_Xi_subsystem, Xi_pure
from modules.functions import *




#%% Parameters

# Initial state
n_qubits = 5
psi0_label = '0' * n_qubits
psi = Statevector.from_label(psi0_label)

# Circuit parameters
Ncliff = 200
seed_list = np.random.randint(0, 1000000, size=(Ncliff, ))
qubits = list(range(n_qubits))


#%% Circuit

clifford_sequence = QuantumCircuit(n_qubits)
magic_circuit = QuantumCircuit(n_qubits)
for i in range(Ncliff):
    clifford = random_clifford_circuit(num_qubits=n_qubits, depth=1, seed=seed_list[i])
    clifford_sequence.compose(clifford, inplace=True)

psi = psi.evolve(clifford_sequence)

l = 2
n = 2
prob_dist = Xi_pure(psi.data)
pauli_strings = []
pauli_iter = product('IXYZ', repeat=l)
for element in pauli_iter:
    pauli_strings.append(''.join(element))

prob_dist_nl = calc_Xi_subsystem(prob_dist, 1, 2, pauli_strings)


#%% Figure
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']
fontsize = 20

fig1 = plt.figure(figsize=(20, 10))
fig1.suptitle(f'Probability distribution n: {n}, l: {l}', fontsize=20)
gs = GridSpec(1, 1, figure=fig1, hspace=0, wspace=0.1)
ax1 = fig1.add_subplot(gs[0, 0])
ax1.plot(np.arange(len(pauli_strings)), prob_dist_nl, 'o', color='blue')
plt.show()