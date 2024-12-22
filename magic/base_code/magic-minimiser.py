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
from modules.InfoLattice import calc_info, plot_info_latt
from modules.MagicLattice import minimal_clifford_disentanglers, minimise_entanglement, calc_classical_magic, \
    calculate_EE_gen, shannon
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
Nlayers = 11
T_per_layer = 2
min_layer, max_layer = 0, 3
seed_list = np.random.randint(0, 1000000, size=(Nlayers, ))
qubits = list(range(n_qubits))
T_layers = np.arange(min_layer, max_layer)
phi_y = np.pi / 2 - np.arctan(1/ np.sqrt(2))
phi_z = np.pi / 4

# Preallocation
info_dict, info_dict_min = {}, {}
SRE1_dict, SRE1_dict_min = {}, {}
EE, SRE1, SRE1_min, NSEE = np.zeros((Nlayers, )), np.zeros((Nlayers, )), np.zeros((Nlayers, )), np.zeros((Nlayers, ))
SRE2, SRE2_min = np.zeros((Nlayers, )), np.zeros((Nlayers, ))
disentanglers = minimal_clifford_disentanglers()
#%% Circuit

clifford_sequence = QuantumCircuit(n_qubits)
magic_circuit = QuantumCircuit(n_qubits)
for i in range(Nlayers):

    loger_main.info(f'Layer: {i}')

    # Generate Clifford layer
    layer = QuantumCircuit(n_qubits)
    # clifford = random_clifford_circuit(num_qubits=n_qubits, depth=20, seed=seed_list[i])
    # clifford_sequence.compose(clifford, inplace=True)
    # layer.compose(clifford, inplace=True)

    # Application of T gates
    rng = np.random.default_rng(seed_list[i])
    rng.shuffle(qubits)
    operands = qubits[-T_per_layer:]
    if min_layer <= i < max_layer:
    #     for qubit in operands:
    #         layer.t(qubit)

        if i==0:
            # layer.ry(phi_y, [1, 2])
            # layer.rz(phi_z, [1, 2])
            layer.h(qubits)
            layer.t([0, 1])
            # layer.t(0)
            # layer.s(qubits)
        if i==1:
            layer.cx(2, 1)
            # layer.t(1)
            layer.cx(1, 0)
            layer.t(0)
            # layer.h(1)

            # layer.t(2)
            # layer.ry(phi_y, [1, 2])

        # if i==2:
            # layer.cx(0, 1)
            # layer.cx(3, 2)
            # layer.cx(1, 2)
            # layer.t(0)
            # layer.t(2)



    magic_circuit.compose(layer, inplace=True)
    magic_circuit.barrier()


    # Circuit evolution
    psi1 = psi1.evolve(layer)
    psi1_min, _ = minimise_entanglement(psi1, n_qubits, disentanglers)
    info_dict[i] = calc_info(psi1.data)
    info_dict_min[i] = calc_info(psi1_min.data)
    SRE1_dict[i] = calc_classical_magic(psi1.data)
    SRE1_dict_min[i] = calc_classical_magic(psi1_min.data)
    SRE1[i] = shannon(psi1.data)
    SRE1_min[i] = shannon(psi1_min.data)
    SRE2[i] = stabiliser_Renyi_entropy_pure(psi1, 2, n_qubits)
    SRE2_min[i] = stabiliser_Renyi_entropy_pure(psi1_min, 2, n_qubits)
    EE[i] = calculate_EE_gen(psi1.data)
    NSEE[i] = calculate_EE_gen(psi1_min.data)



if not np.allclose(SRE1, SRE1_min):
    raise ValueError('SRE1 changing under minimisation')
if not np.allclose(SRE2, SRE2_min):
    raise ValueError('SRE2 changing under minimisation')
magic_circuit.draw(output="mpl", style="iqp")
#%% Figure format
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
colormap = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=color_map)

# Defining a colormap with negatives
color_map_neg = plt.get_cmap('PuOr').reversed()
colors_neg = color_map_neg(np.linspace(0, 1, 41))
colors_neg[20] = [1, 1, 1, 1]
color_map_neg = LinearSegmentedColormap.from_list("custom_colormap", colors_neg)
colormap_neg = cm.ScalarMappable(norm=Normalize(vmin=-2, vmax=2), cmap=color_map_neg)


#%% Figure 1: Information Lattice

Ncol = int(Nlayers / 2) + 1
Nrows = (Nlayers // 10 + 1)
fig1 = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 1, figure=fig1, hspace=0.5)
sub_gs0 = gs[0].subgridspec(Nrows, Ncol + 1, hspace=0.2, wspace=0.2)
sub_gs1 = gs[1].subgridspec(Nrows, Ncol + 1, hspace=0.2, wspace=0.2)

fig1.suptitle('Information lattice', fontsize=20)
fig1.text(0.1, 0.7, 'Not minimised', fontsize=20, rotation=90)
fig1.text(0.1, 0.2, 'Minimised', fontsize=20, rotation=90)

for i in range(Nlayers):
    row = i // Ncol
    col = i % Ncol
    ax = fig1.add_subplot(sub_gs0[row, col])
    ax1 = fig1.add_subplot(sub_gs1[row, col])
    plot_info_latt(info_dict[i], ax, color_map, max_value=max_value, indicate_ints=True)
    plot_info_latt(info_dict_min[i], ax1, color_map, max_value=max_value, indicate_ints=True)
    ax.set_title(f'$N= {i}$')
    ax1.set_title(f'$N= {i}$')

# Fig 1: Colorbar
cbar_ax = fig1.add_subplot(sub_gs0[:, -1])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("left", size="20%", pad=0)
cbar = fig1.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()
cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)


#%% Figure 2: SRE1 Lattice

Ncol = int(Nlayers / 2) + 1
Nrows = (Nlayers // 10 + 1)
fig2 = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 1, figure=fig2, hspace=0.5)
sub_gs0 = gs[0].subgridspec(Nrows, Ncol + 1, hspace=0.2, wspace=0.2)
sub_gs1 = gs[1].subgridspec(Nrows, Ncol + 1, hspace=0.2, wspace=0.2)

fig2.suptitle('SRE1 information lattice', fontsize=20)
fig2.text(0.1, 0.7, 'Not minimised', fontsize=20, rotation=90)
fig2.text(0.1, 0.2, 'Minimised', fontsize=20, rotation=90)

for i in range(Nlayers):
    row = i // Ncol
    col = i % Ncol
    ax = fig2.add_subplot(sub_gs0[row, col])
    ax1 = fig2.add_subplot(sub_gs1[row, col])
    plot_info_latt(SRE1_dict[i], ax, color_map, max_value=max_value, indicate_ints=True)
    plot_info_latt(SRE1_dict_min[i], ax1, color_map, max_value=max_value, indicate_ints=True)
    ax.set_title(f'$N= {i}$')
    ax1.set_title(f'$N= {i}$')

# Fig 1: Colorbar
cbar_ax = fig2.add_subplot(sub_gs0[:, -1])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("left", size="20%", pad=0)
cbar = fig2.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()
cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)




#%% Figure 3: Entropies

fig3 = plt.figure()
gs = GridSpec(1, 1, figure=fig3)
ax = fig3.add_subplot(gs[0, 0])

ax.plot(np.linspace(0, Nlayers - 1, Nlayers), SRE1, alpha=0.5, color=color_list[0], marker='o', label='SRE1')
ax.plot(np.linspace(0, Nlayers - 1, Nlayers), SRE2, alpha=0.5, color=color_list[1], marker='o', label='SRE2')
ax.plot(np.linspace(0, Nlayers - 1, Nlayers), EE, alpha=0.5, color=color_list[2], marker='o', label='EE cuts')
ax.plot(np.linspace(0, Nlayers - 1, Nlayers), NSEE, alpha=0.5, color=color_list[3], marker='o', label='NSEE')

ax.plot(T_layers, SRE1[T_layers], marker='o', linestyle='None', markerfacecolor='None', markeredgecolor='k')
ax.plot(T_layers, SRE2[T_layers], marker='o', linestyle='None', markerfacecolor='None', markeredgecolor='k')
ax.plot(T_layers, EE[T_layers], marker='o', linestyle='None', markerfacecolor='None', markeredgecolor='k')
ax.plot(T_layers, NSEE[T_layers], marker='o', linestyle='None', markerfacecolor='None', markeredgecolor='k')

ax.legend(loc='best', ncol=4, frameon=False, fontsize=10)
ax.set_xlabel('Circuit layer', fontsize=16)
ax.set_ylabel('Magic and entanglement', fontsize=16)
ax.set_xlim([0, Nlayers])
ax.set_ylim([-0.1, np.max(EE) + 1])

plt.show()





