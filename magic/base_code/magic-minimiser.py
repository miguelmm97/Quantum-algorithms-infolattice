#%% Imports

# Managing data
import os
import sys
import h5py
from datetime import date

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace

# Information lattice and functions
from modules.InfoLattice import calc_info
from modules.MagicLattice import minimal_clifford_disentanglers, minimise_entanglement, calc_magic, calculate_EE_gen
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
n_qubits = 5
psi0_label = '0' * n_qubits
psi1 = Statevector.from_label(psi0_label)

# Circuit parameters
Nlayers = 20
T_per_layer = 1
min_layer, max_layer = 0, 20
seed_list = np.random.randint(0, 1000000, size=(Nlayers, ))
qubits = list(range(n_qubits))

# Preallocation
info_dict, info_dict_min = {}, {}
SRE1_dict, SRE1_dict_min = {}, {}
EE, SRE, SRE_min, NSEE = np.zeros((Nlayers, )), np.zeros((Nlayers, )), np.zeros((Nlayers, )), np.zeros((Nlayers, ))
disentanglers = minimal_clifford_disentanglers()
#%% Circuit

clifford_sequence = QuantumCircuit(n_qubits)
magic_circuit = QuantumCircuit(n_qubits)
for i in range(Nlayers):

    loger_main.info(f'Layer: {i}')

    # Generate Clifford layer
    layer = QuantumCircuit(n_qubits)
    clifford = random_clifford_circuit(num_qubits=n_qubits, depth=50, seed=seed_list[i])
    clifford_sequence.compose(clifford, inplace=True)
    layer.compose(clifford, inplace=True)

    # Application of T gates
    rng = np.random.default_rng(seed_list[i])
    rng.shuffle(qubits)
    operands = qubits[-T_per_layer:]
    if min_layer < i < max_layer:
        for qubit in operands:
            layer.t(qubit)
    magic_circuit.compose(layer, inplace=True)


    # Circuit evolution
    psi1 = psi1.evolve(layer)
    psi1_min, _ = minimise_entanglement(psi1, n_qubits, disentanglers)
    info_dict[i] = calc_info(psi1.data)
    info_dict_min[i] = calc_info(psi1_min.data)
    SRE1_dict[i] = calc_magic(psi1.data)
    SRE1_dict_min[i] = calc_magic(psi1_min.data)
    SRE[i] = stabiliser_Renyi_entropy_pure(psi1, 2, n_qubits)
    SRE_min[i] = stabiliser_Renyi_entropy_pure(psi1_min, 2, n_qubits)
    EE[i] = calculate_EE_gen(psi1)
    NSEE[i] = calculate_EE_gen(psi1_min)


#%% Figure
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']


# Defining a colormap without negatives
color_map = plt.get_cmap("PuOr").reversed()
colors = color_map(np.linspace(0, 1, 41)[20:])
colors[0] = [1, 1, 1, 1]
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)

# Normalising the colormap
max_value = 0.
for key1 in info_dict.keys():
    for key2 in info_dict[key1].keys():
        value = max(info_dict[key1][key2])
        if value > max_value:
            max_value = value
colormap = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=2), cmap=color_map)



# Defining a colormap with negatives
color_map_neg = plt.get_cmap('PuOr').reversed()
colors_neg = color_map_neg(np.linspace(0, 1, 41))
colors_neg[20] = [1, 1, 1, 1]
color_map_neg = LinearSegmentedColormap.from_list("custom_colormap", colors_neg)
colormap_neg = cm.ScalarMappable(norm=Normalize(vmin=-2, vmax=2), cmap=color_map_neg)



# Fig 1: Plot
Nrows = int(Nlayers / (info_interval * 10))
Ncol = 10
fig1 = plt.figure(figsize=(20, 10))
fig1.suptitle('Info lattice', fontsize=20)
gs = GridSpec(Nrows, Ncol+1, figure=fig1, hspace=0, wspace=0.1)
for i in range(int(Nlayers / info_interval)):
    # Position in the grid
    row = i // Ncol
    col = i % Ncol
    ax = fig1.add_subplot(gs[row, col])
    # Plots
    plot_info_latt(info_dict[i], ax, color_map, max_value=max_value, indicate_ints=True)
    if min_block < i < max_block:
        ax.set_title(f'$N= {i * int(Nlayers / Nblocks)}$, $T + C$')
    else:
        ax.set_title(f'$N= {i * int(Nlayers / Nblocks)}$, $C$')



# Fig 1: Colorbar
cbar_ax = fig1.add_subplot(gs[:, -1])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("left", size="20%", pad=0)
cbar = fig1.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()
cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)





