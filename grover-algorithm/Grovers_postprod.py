#%% Modules setup

# Math and plotting
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn
import itertools

# Managing system, data and config files
from functions import load_my_data, load_my_attr

#%% Loading data
file_list = ['Experiment1.h5', 'Experiment2.h5', 'Experiment3.h5', 'Experiment4.h5', 'Experiment5.h5',  'Experiment6.h5']
data_dict = load_my_data(file_list, '../Data')

#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
palette1 = seaborn.color_palette(palette='Blues', n_colors=len(file_list))
palette2 = seaborn.color_palette(palette='dark:salmon_r', n_colors=len(file_list))
markers = itertools.cycle(('d', 's', '.', 'o', '*'))


# Statistics
fig1 = plt.figure(figsize=(9, 6))
ax1 = fig1.gca()
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 0.06)
ax1.set_xlabel("$t/t_{opt}$", fontsize=20)
ax1.set_ylabel("$\\langle l \\rangle$", fontsize=20)
ax1.tick_params(which='major', width=0.75, labelsize=15)
ax1.tick_params(which='major', length=10, labelsize=15)
ax1.set_title('Expected value of the scale of information')

fig2 = plt.figure(figsize=(9, 6))
ax2 = fig2.gca()
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 0.06)
ax2.set_xlabel("$t/t_{opt}$", fontsize=20)
ax2.set_ylabel("$\\langle l \\rangle$", fontsize=20)
ax2.tick_params(which='major', width=0.75, labelsize=15)
ax2.tick_params(which='major', length=10, labelsize=15)
ax2.set_title('Median of the scale of information')


for i, key in enumerate(data_dict.keys()):
    num_qubits     = data_dict[key]['Parameters']['num_qubits']
    marked_state   = data_dict[key]['Parameters']['marked_state_str']
    n_iter         = data_dict[key]['Parameters']['n_iter']
    optimal_iter   = data_dict[key]['Parameters']['opt_iter']
    l_rescaled     = np.arange(0, num_qubits) / (num_qubits - 1)
    info_per_scale = data_dict[key]['Simulation']['info_per_scale']
    mean_info      = data_dict[key]['Simulation']['mean_info']
    t_rescaled     = np.arange(0, n_iter + 1) / optimal_iter
    marker = next(markers)

    # Expected value
    ax1.plot(t_rescaled, mean_info, marker=marker, color=palette1[i], label=f'$n=$ {num_qubits}, $|\psi_m\\rangle=$ {marked_state}')
    ax1.plot(t_rescaled, mean_info, color=palette1[i])

    ax2.plot(t_rescaled, median_info, marker=marker, color=palette1[i],
             label=f'$n=$ {num_qubits}, $|\psi_m\\rangle=$ {marked_state}')
    ax2.plot(t_rescaled, median_info, color=palette1[i])

ax1.legend(loc='best', fontsize=10, frameon=False)
plt.show()