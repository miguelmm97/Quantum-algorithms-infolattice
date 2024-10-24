#%% modules setup

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Modules
from modules.functions import *

#%% Loading data
file_list = ['Exp6.h5', 'Exp7.h5', 'Exp9.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/quantum-algorithms-info/git-repo/magic/data')


# Parameters
n_qubits1      = data_dict[file_list[0]]['Parameters']['n_qubits']
Nlayers1       = data_dict[file_list[0]]['Parameters']['Nlayers']
Nblocks1       = data_dict[file_list[0]]['Parameters']['Nblocks']
T_per_block1   = data_dict[file_list[0]]['Parameters']['T_per_block']
min_block1     = data_dict[file_list[0]]['Parameters']['min_block']
max_block1     = data_dict[file_list[0]]['Parameters']['max_block']

n_qubits2      = data_dict[file_list[1]]['Parameters']['n_qubits']
Nlayers2       = data_dict[file_list[1]]['Parameters']['Nlayers']
Nblocks2       = data_dict[file_list[1]]['Parameters']['Nblocks']
T_per_block2   = data_dict[file_list[1]]['Parameters']['T_per_block']
min_block2     = data_dict[file_list[1]]['Parameters']['min_block']
max_block2     = data_dict[file_list[1]]['Parameters']['max_block']

n_qubits3      = data_dict[file_list[2]]['Parameters']['n_qubits']
Nlayers3       = data_dict[file_list[2]]['Parameters']['Nlayers']
Nblocks3       = data_dict[file_list[2]]['Parameters']['Nblocks']
T_per_block3   = data_dict[file_list[2]]['Parameters']['T_per_block']
min_block3     = data_dict[file_list[2]]['Parameters']['min_block']
max_block3     = data_dict[file_list[2]]['Parameters']['max_block']

# Simulation data
seed_list1         = data_dict[file_list[0]]['Simulation']['seed']
magic1             = data_dict[file_list[0]]['Simulation']['magic']
SRE1               = data_dict[file_list[0]]['Simulation']['SRE']
SRE_long1          = data_dict[file_list[0]]['Simulation']['SRE_long']
SRE_clifford1      = data_dict[file_list[0]]['Simulation']['SRE_clifford']
SRE_long_clifford1 = data_dict[file_list[0]]['Simulation']['SRE_long_clifford']

seed_list2         = data_dict[file_list[1]]['Simulation']['seed']
magic2             = data_dict[file_list[1]]['Simulation']['magic']
SRE2               = data_dict[file_list[1]]['Simulation']['SRE']
SRE_long2          = data_dict[file_list[1]]['Simulation']['SRE_long']
SRE_clifford2      = data_dict[file_list[1]]['Simulation']['SRE_clifford']
SRE_long_clifford2 = data_dict[file_list[1]]['Simulation']['SRE_long_clifford']

seed_list3         = data_dict[file_list[2]]['Simulation']['seed']
magic3             = data_dict[file_list[2]]['Simulation']['magic']
SRE3               = data_dict[file_list[2]]['Simulation']['SRE']
SRE_long3          = data_dict[file_list[2]]['Simulation']['SRE_long']
SRE_clifford3      = data_dict[file_list[2]]['Simulation']['SRE_clifford']
SRE_long_clifford3 = data_dict[file_list[2]]['Simulation']['SRE_long_clifford']


# Plot variables
label1 = f'$N={Nlayers1}$ , $n_T= {T_per_block1},$' + ' min: ' + f'${min_block1}$,' + ' max: ' + f'${max_block1}$'
T_layers1 = int(Nlayers1/ Nblocks1) * np.arange(min_block1, max_block1)

label2 = f'$N={Nlayers2}$ , $n_T= {T_per_block2},$' + ' min: ' + f'${min_block2}$,' + ' max: ' + f'${max_block2}$'
T_layers2 = int(Nlayers2/ Nblocks2) * np.arange(min_block2, max_block2)

label3 = f'$N={Nlayers3}$ , $n_T= {T_per_block3},$' + ' min: ' + f'${min_block3}$,' + ' max: ' + f'${max_block3}$'
T_layers3 = int(Nlayers3/ Nblocks3) * np.arange(min_block3, max_block3)

#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
line_list = ['solid', 'dashed', 'dashdot', 'dotted']
markersize = 5
fontsize = 20

# Fig 1: Definition
fig1 = plt.figure(figsize=(20, 10))
gs = GridSpec(1, 1, figure=fig1, hspace=0, wspace=0)
ax = fig1.add_subplot(gs[0, 0])

# Fig 1: Plots
ax.plot(np.linspace(0, Nlayers1 - 1, Nlayers1), magic1, alpha=0.5, color=color_list[0], label=label1)
ax.plot(np.linspace(0, Nlayers1 - 1, Nlayers1), magic1, marker=marker_list[0], color=color_list[0], linestyle='None')
ax.plot(T_layers1, magic1[T_layers1], marker=marker_list[0], linestyle='None', markerfacecolor='None', markeredgecolor='k')

ax.plot(np.linspace(0, Nlayers2 - 1, Nlayers2), magic2, alpha=0.5, color=color_list[1], label=label2)
ax.plot(np.linspace(0, Nlayers2 - 1, Nlayers2), magic2, marker=marker_list[0], color=color_list[1], linestyle='None')
ax.plot(T_layers2, magic2[T_layers2], marker=marker_list[0], linestyle='None', markerfacecolor='None', markeredgecolor='k')

ax.plot(np.linspace(0, Nlayers3 - 1, Nlayers3), magic3, alpha=0.5, color=color_list[2], label=label3)
ax.plot(np.linspace(0, Nlayers3 - 1, Nlayers3), magic3, marker=marker_list[0], color=color_list[2], linestyle='None')
ax.plot(T_layers3, magic3[T_layers3], marker=marker_list[0], linestyle='None', markerfacecolor='None', markeredgecolor='k')
ax.legend(loc='upper center', ncol=2, frameon=False, fontsize=16,  bbox_to_anchor=(0.5, 1.15))

# Fig 1: Format
x_axis_ticks = [int(i) for i in np.linspace(0, Nlayers1, 10)]
x_axis_labels = [str(int(i)) for i in np.linspace(0, Nlayers1, 10)]
ax.set(xticks=x_axis_ticks, xticklabels=x_axis_labels)
ax.set_ylim(0, np.max(magic1))

ax.set_xlabel('$N_l$', fontsize=fontsize)
ax.set_ylabel('Non-integer magic', fontsize=fontsize)
ax.tick_params(which='major', width=0.75, labelsize=fontsize)
ax.tick_params(which='major', length=6, labelsize=fontsize)


fig1.savefig(f'../{file_list[0]}.pdf', format='pdf', backend='pgf')
plt.show()
