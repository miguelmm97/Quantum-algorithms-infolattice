#%% modules setup

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Modules
from modules.functions import *

#%% Loading data
file_list = ['Exp1.h5']
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
magic             = data_dict[file_list[0]]['Simulation']['magic']
SRE               = data_dict[file_list[0]]['Simulation']['SRE']
SRE_long          = data_dict[file_list[0]]['Simulation']['SRE_long']
SRE_clifford      = data_dict[file_list[0]]['Simulation']['SRE_clifford']
SRE_long_clifford = data_dict[file_list[0]]['Simulation']['SRE_long_clifford']

# Plot variables
label1 = f'$N={Nlayers}$ , $n_T= {T_per_block},$' + ' min: ' + f'${min_block}$,' + ' max: ' + f'${max_block}$'
T_layers = int(Nlayers/ Nblocks) * np.arange(min_block, max_block)

#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
line_list = ['solid', 'dashed', 'dashdot', 'dotted']
markersize = 5
fontsize=20

# Fig 1: Definition
fig1 = plt.figure(figsize=(15, 8))
gs = GridSpec(1, 1, figure=fig1, hspace=0, wspace=0)
ax = fig1.add_subplot(gs[0, 0])

# Fig 1: Plots
ax.plot(np.linspace(0, Nlayers - 1, Nlayers), magic, alpha=0.5, color=color_list[0], label=label1)
ax.plot(np.linspace(0, Nlayers - 1, Nlayers), magic, marker=marker_list[0], color=color_list[0], linestyle='None')
ax.plot(T_layers, magic[T_layers], marker=marker_list[0], linestyle='None', markerfacecolor='None', markeredgecolor='k')
ax.legend(ncol=1, frameon=False, fontsize=16)

# Fig 1: Format
x_axis_ticks = [int(i) for i in np.linspace(0, Nlayers, 10)]
x_axis_labels = [str(int(i)) for i in np.linspace(0, Nlayers, 10)]
ax.set(xticks=x_axis_ticks, xticklabels=x_axis_labels)
ax.set_ylim(0, np.max(magic))

ax.set_xlabel('$N_l$', fontsize=fontsize)
ax.set_ylabel('Non-integer magic', fontsize=fontsize)
ax.tick_params(which='major', width=0.75, labelsize=fontsize)
ax.tick_params(which='major', length=6, labelsize=fontsize)
plt.show()
