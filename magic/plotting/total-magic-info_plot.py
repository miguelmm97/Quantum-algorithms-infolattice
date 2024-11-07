#%% modules setup

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Modules
from modules.functions import *

#%% Loading data
file_list = ['Exp22.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/quantum-algorithms-info/git-repo/magic/data') # '../data'


# Parameters
n_qubits1      = data_dict[file_list[0]]['Parameters']['n_qubits']
Nlayers1       = data_dict[file_list[0]]['Parameters']['Nlayers']
Nblocks1       = data_dict[file_list[0]]['Parameters']['Nblocks']
T_per_block1   = data_dict[file_list[0]]['Parameters']['T_per_block']
min_block1     = data_dict[file_list[0]]['Parameters']['min_block']
max_block1     = data_dict[file_list[0]]['Parameters']['max_block']

# Simulation data
total_magic_info  = data_dict[file_list[0]]['Simulation']['total_magic_info']
SRE               = data_dict[file_list[0]]['Simulation']['SRE']
shannon           = data_dict[file_list[0]]['Simulation']['shannon']
T_layers1 = int(Nlayers1/ Nblocks1) * np.arange(min_block1, max_block1)


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
ax.plot(np.linspace(0, Nlayers1 - 1, Nlayers1), total_magic_info, alpha=0.5, color=color_list[0], label='$I$')
ax.plot(np.linspace(0, Nlayers1 - 1, Nlayers1), SRE, alpha=0.5, color=color_list[1], label='$M_2$')
ax.plot(np.linspace(0, Nlayers1 - 1, Nlayers1), shannon, alpha=0.5, color=color_list[2], label='$M_1$')
ax.plot(T_layers1, total_magic_info[T_layers1], marker=marker_list[0], linestyle='None', markerfacecolor='None', markeredgecolor='k')
ax.legend(loc='best', ncol=1, frameon=False, fontsize=16)

# Fig 1: Format
x_axis_ticks = [int(i) for i in np.linspace(0, Nlayers1, 10)]
x_axis_labels = [str(int(i)) for i in np.linspace(0, Nlayers1, 10)]
ax.set(xticks=x_axis_ticks, xticklabels=x_axis_labels)
ax.set_ylim(0, np.max(total_magic_info + 1))

ax.set_xlabel('$N_l$', fontsize=fontsize)
ax.set_ylabel('Total magic info', fontsize=fontsize)
ax.tick_params(which='major', width=0.75, labelsize=fontsize)
ax.tick_params(which='major', length=6, labelsize=fontsize)

fig1.savefig(f'../{file_list[0]}.pdf', format='pdf', backend='pgf')
plt.show()
