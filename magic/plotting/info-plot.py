#%% modules setup

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Modules
from modules.functions import *
from modules.InfoLattice import calc_info, plot_info_latt
#%% Loading data
file_list = ['Exp4.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/quantum-algorithms-info/git-repo/magic/data')


# Parameters
n_qubits      = data_dict[file_list[0]]['Parameters']['n_qubits']
Nlayers       = data_dict[file_list[0]]['Parameters']['Nlayers']
Nblocks       = data_dict[file_list[0]]['Parameters']['Nblocks']
T_per_block   = data_dict[file_list[0]]['Parameters']['T_per_block']
min_block     = data_dict[file_list[0]]['Parameters']['min_block']
max_block     = data_dict[file_list[0]]['Parameters']['max_block']

# Simulation data
seed_list         = data_dict[file_list[0]]['Simulation']['seed']

# Plot title string
label1 = f'$N={Nlayers}$ , $n_T= {T_per_block},$' + ' min: ' + f'${min_block}$,' + ' max: ' + f'${max_block}$'

#%% Reconstruction of the info lattice

# Parameters
psi0_label = '0' * n_qubits
psi1 = Statevector.from_label(psi0_label)
psi2 = Statevector.from_label(psi0_label)
info_interval = int(Nlayers/ Nblocks)
qubits = list(range(n_qubits))
info_dict, info_dict_clifford = {}, {}

# Circuit evolution
clifford_sequence = QuantumCircuit(n_qubits)
magic_circuit = QuantumCircuit(n_qubits)
for i in range(Nlayers):

    # Generate Clifford layer
    layer = QuantumCircuit(n_qubits)
    clifford = random_clifford_circuit(num_qubits=n_qubits, depth=1, seed=seed_list[i])
    clifford_sequence.compose(clifford, inplace=True)
    layer.compose(clifford, inplace=True)

    # Application of T gates
    rng = np.random.default_rng(seed_list[i])
    rng.shuffle(qubits)
    operands = qubits[-T_per_block:]
    if (i % info_interval) == 0 and min_block < (i // info_interval) < max_block:
        for qubit in operands:
            layer.t(qubit)
    magic_circuit.compose(layer, inplace=True)

    # Information lattice and magic measures
    psi1 = psi1.evolve(layer)
    psi2 = psi2.evolve(clifford)
    if (i % info_interval) == 0:
        info_dict[i // info_interval] = calc_info(psi1.data)
        info_dict_clifford[i // info_interval] = calc_info(psi2.data)


#%% Figure
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']

# Defining a colormap (starting in white)
color_map = plt.get_cmap("Oranges")
colors = color_map(np.linspace(0, 1, 20))
colors[0] = [1, 1, 1, 1]
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)

# Normalising the colormap
max_value = 0.
for key1 in info_dict_clifford.keys():
    for key2 in info_dict_clifford[key1].keys():
        value = max(info_dict_clifford[key1][key2])
        if value > max_value:
            max_value = value
colormap = cm.ScalarMappable(norm=Normalize(vmin=0., vmax=max_value), cmap=color_map)

# Fig 1: Plot
Nrows = int(Nlayers / (info_interval * 10))
Ncol = 10
fig1 = plt.figure(figsize=(20, 10))
fig1.suptitle('$\mathcal{U}\\vert \psi \\rangle$ with $\mathcal{U}=\langle H, S, CNOT, T\\rangle$', fontsize=20)
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


# Fig 2: Plots
fig2 = plt.figure(figsize=(20, 10))
fig2.suptitle('$\mathcal{U}\\vert \psi \\rangle$ with $\mathcal{U}=\langle H, S, CNOT\\rangle$', fontsize=20)
gs = GridSpec(Nrows, Ncol + 1, figure=fig2, hspace=0, wspace=0.1)
for i in range(int(Nlayers/ info_interval)):
    # Position in the grid
    row = i // Ncol
    col = i % Ncol
    ax = fig2.add_subplot(gs[row, col])
    # Plots
    plot_info_latt(info_dict_clifford[i], ax, color_map, max_value=max_value, indicate_ints=True)
    ax.set_title(f'$N= {i * int(Nlayers/ Nblocks)}$')


# Fig 2: Colorbar
cbar_ax = fig2.add_subplot(gs[:, -1])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("left", size="20%", pad=0)
cbar = fig2.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()
cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)

plt.show()

