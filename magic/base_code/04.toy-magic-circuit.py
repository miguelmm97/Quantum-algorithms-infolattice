#%% Imports

# Built-in modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import HGate, CXGate, SGate

# Information lattice
from modules.InfoLattice import *
from modules.functions import *
from modules.MagicLattice import *
#%% Main

n_qubits = 4
psi0_label = '0' * n_qubits
psi0 = Statevector.from_label(psi0_label)
circuit_list = []

circuit_list.append(HGate(0))
circuit_list.append(SGate(1))
circuit_list.append(CXGate(0, 1))


font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']
fontsize = 20

fig1 = plt.figure()
plot_lattice_from_circuit(circuit_list, psi0, fig1)
plt.show()