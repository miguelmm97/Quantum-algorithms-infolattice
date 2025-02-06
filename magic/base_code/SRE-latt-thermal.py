#%% Imports

# Managing data
import os
import sys
import h5py
from datetime import date

# Imports from Qiskit
from qiskit import QuantumCircuit

# Information lattice and functions
from modules.InfoLattice import plot_info_latt, calc_info
from modules.MagicLattice import calc_classical_magic, minimise_entanglement, minimal_clifford_disentanglers
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


#%% Parameters

# Initial state
n_qubits = 6
psi0_label = '0' * n_qubits
psi1 = Statevector.from_label(psi0_label)

# Circuit parameters
Nlayers = 100
T_per_layer = 1
min_layers, max_layers = 0, 0
seed_list = np.random.randint(0, 1000000, size=(Nlayers, ))
qubits = list(range(n_qubits))
disentanglers = minimal_clifford_disentanglers()

#%% Circuit

clifford_sequence = QuantumCircuit(n_qubits)
random_circuit = QuantumCircuit(n_qubits)
for i in range(Nlayers):
    loger_main.info(f'Layer: {i}')

    # Generate Clifford layer
    layer = QuantumCircuit(n_qubits)
    clifford = random_clifford_circuit(num_qubits=n_qubits, depth=10, seed=seed_list[i])
    clifford_sequence.compose(clifford, inplace=True)
    layer.compose(clifford, inplace=True)

    # Application of T gates
    if min_layers <= i < max_layers:
        rng = np.random.default_rng(seed_list[i])
        rng.shuffle(qubits)
        operands = qubits[-T_per_layer:]
        for qubit in operands:
            layer.t(qubit)
    random_circuit.compose(layer, inplace=True)

# Circuit evolution: SRE1
psi = psi1.evolve(random_circuit)
SRE1_infolatt = calc_classical_magic(psi.data)
VN_infolatt = calc_info(psi.data)

# Circuit evolution: Entanglement entropy
psi_min, _ = minimise_entanglement(psi, n_qubits, disentanglers)
SRE1_infolatt_min = calc_classical_magic(psi_min.data)
VN_infolatt_min = calc_info(psi_min.data)


#%% Figure

# Format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']

# Defining a colormap without negatives and white color for no info
color_map = plt.get_cmap("PuOr").reversed()
colors = color_map(np.linspace(0, 1, 41)[20:])
colors[0] = [1, 1, 1, 1]
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)      # Colormap for the info lattice
colormap = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=2), cmap=color_map)  # Colormap for the colorbar




# SRE1 information lattice
fig1 = plt.figure()
gs = GridSpec(1, 2, figure=fig1, hspace=0.5)
ax0 = fig1.add_subplot(gs[0, 0])
ax1 = fig1.add_subplot(gs[0, 1])

fig1.suptitle('SRE1 information lattice', fontsize=20)
plot_info_latt(SRE1_infolatt, ax0, color_map, indicate_ints=True)
plot_info_latt(SRE1_infolatt_min, ax1, color_map, indicate_ints=True)
ax0.set_title(f'Not minimised')
ax1.set_title(f'Minimised')

cbar_ax = fig1.add_subplot(gs[-1, :])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("bottom", size="5%", pad=0)
cbar = fig1.colorbar(colormap, cax=cax, orientation='horizontal')
cbar_ax.set_axis_off()
cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)




# Von Neumann information lattice
fig2 = plt.figure()
gs = GridSpec(1, 2, figure=fig2, hspace=0.5)
ax0 = fig2.add_subplot(gs[0, 0])
ax1 = fig2.add_subplot(gs[0, 1])

fig2.suptitle('Von Neumann information lattice', fontsize=20)
plot_info_latt(VN_infolatt, ax0, color_map, indicate_ints=True)
plot_info_latt(VN_infolatt_min, ax1, color_map, indicate_ints=True)
ax0.set_title(f'Not minimised')
ax1.set_title(f'Minimised')

# Fig 1: Colorbar
cbar_ax = fig2.add_subplot(gs[-1, :])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("bottom", size="5%", pad=0)
cbar = fig2.colorbar(colormap, cax=cax, orientation='horizontal')
cbar_ax.set_axis_off()
cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)
plt.show()



