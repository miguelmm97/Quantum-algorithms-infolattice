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

# Information lattice
from modules.InfoLattice import calc_info, plot_info_latt
from modules.functions import random_wn_state
from modules.MagicLattice import calc_classical_magic, plot_magic_latt, calc_all_Xi, plot_probabilities
from code_alma import find_best_circuit_old


#%% Main


# W(n, k) state
n = 3
k = 3
psi_w = random_wn_state(n, k)
info_latt = calc_info(psi_w)
SRE1_latt = calc_classical_magic(psi_w)
magic_latt = {key: info_latt[key] - SRE1_latt[key] for key in info_latt.keys()}


# W(n, k) circuit
info_dict = {}
SRE1_dict = {}
magic_dict = {}
prob_dist = {}
n_qubits = 4
qc_w = QuantumCircuit(n_qubits)
psi0_label = '0' * n_qubits
psi0 = Statevector.from_label(psi0_label)
phi_y = np.pi / 2 - np.arctan(1/ np.sqrt(2))
phi_z = np.pi / 4

# qc_w.ry(phi_y, 0)
# qc_w.ry(phi_y, 2)
qc_w.h([0, 2])
qc_w.cx(0, 1)
qc_w.cx(2, 1)
qc_w.h(3)
qc_w.t(3)



psi1 = psi0.evolve(qc_w)
min_qc = find_best_circuit_old(psi1, n_qubits)[-1]
psi = psi1.evolve(min_qc)
info_dict[0] = calc_info(psi0.evolve(qc_w).data)
SRE1_dict[0] = calc_classical_magic(psi.data)
magic_dict[0] = {key: info_dict[0][key] - SRE1_dict[0][key] for key in info_dict[0].keys()}
# prob_dist[0] = calc_all_Xi(psi1.evolve(qc_w).data)


# qc_w.rz(phi_z, 0)
qc_w.h(0)
qc_w.cx(0, 1)
psi1 = psi0.evolve(qc_w)
min_qc = find_best_circuit_old(psi1, n_qubits)[-1]
psi = psi1.evolve(min_qc)
info_dict[1] = calc_info(psi0.evolve(qc_w).data)
SRE1_dict[1] = calc_classical_magic(psi.data)
magic_dict[1] = {key: info_dict[1][key] - SRE1_dict[1][key] for key in info_dict[1].keys()}
# prob_dist[1] = calc_all_Xi(psi1.evolve(qc_w).data)

# qc_w.rz(phi_z * 2, 0)
# qc_w.ry(phi_y, 0)
# qc_w.rz(phi_z, 0)
qc_w.s(2)
qc_w.cx(2, 1)
psi1 = psi0.evolve(qc_w)
min_qc = find_best_circuit_old(psi1, n_qubits)[-1]
psi = psi1.evolve(min_qc)
info_dict[2] =  calc_info(psi0.evolve(qc_w).data)
SRE1_dict[2] = calc_classical_magic(psi.data)
magic_dict[2] = {key: info_dict[2][key] - SRE1_dict[2][key] for key in info_dict[2].keys()}
# prob_dist[2] = calc_all_Xi(psi1.evolve(qc_w).data)

# qc_w.ry(phi_y, 1)
# qc_w.rz(phi_z  *3, 1)
qc_w.h(1)
qc_w.cx(1, 0)
# qc_w.cx(1, 2)
psi = psi0.evolve(qc_w)
# min_qc = find_best_circuit_old(psi1, n_qubits)[-1]
# psi = psi1.evolve(min_qc)
info_dict[3] =  calc_info(psi0.evolve(qc_w).data)
SRE1_dict[3] = calc_classical_magic(psi.data)
magic_dict[3] = {key: info_dict[3][key] - SRE1_dict[3][key] for key in info_dict[3].keys()}
# prob_dist[3] = calc_all_Xi(psi1.evolve(qc_w).data)

# qc_w.ry(phi_y, 1)
# qc_w.rz(phi_z, 1)
# qc_w.ry(phi_y, 0).inverse()
# qc_w.ry(phi_y, 2).inverse()

psi = psi0.evolve(qc_w)
# min_qc = find_best_circuit_old(psi1, n_qubits)[-1]
# psi = psi1.evolve(min_qc)
info_dict[4] = calc_info(psi.data)
SRE1_dict[4] = calc_classical_magic(psi.data)
magic_dict[4] = {key: info_dict[4][key] - SRE1_dict[4][key] for key in info_dict[4].keys()}
# prob_dist[4] = calc_all_Xi(psi1.evolve(qc_w).data)

qc_w.draw(output="mpl", style="iqp")







#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']
fontsize = 20

# Defining a colormap without negatives
color_map = plt.get_cmap("PuOr").reversed()
colors = color_map(np.linspace(0, 1, 41)[20:])
colors[0] = [1, 1, 1, 1]
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)

# Normalising the colormap
max_value = 2.
colormap = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=2), cmap=color_map)

# Defining a colormap with negatives
color_map_neg = plt.get_cmap('PuOr').reversed()
colors_neg = color_map_neg(np.linspace(0, 1, 41))
colors_neg[20] = [1, 1, 1, 1]
color_map_neg = LinearSegmentedColormap.from_list("custom_colormap", colors_neg)
colormap_neg = cm.ScalarMappable(norm=Normalize(vmin=-2, vmax=2), cmap=color_map_neg)



# # Information
# fig1 = plt.figure()
# gs = GridSpec(1, 1, figure=fig1, hspace=0, wspace=0.1)
# ax = fig1.add_subplot(gs[0, 0])
# plot_info_latt(info_latt, ax, color_map, max_value=max_value, indicate_ints=True)
# ax.set_title('Information lattice')
#
# cbar_ax = fig1.gca()
# divider = make_axes_locatable(cbar_ax)
# cax = divider.append_axes("right", size="5%", pad=0.3)
# cbar = fig1.colorbar(colormap, cax=cax, orientation='vertical')
# cbar_ax.set_axis_off()
# cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)
#
#
# # SRE1 lattice
# fig2 = plt.figure()
# gs = GridSpec(1, 1, figure=fig2, hspace=0, wspace=0.1)
# ax = fig2.add_subplot(gs[0, 0])
# plot_magic_latt(SRE1_latt, ax, color_map, max_value=max_value, indicate_ints=True)
# ax.set_title('SRE1 lattice')
#
#
# cbar_ax = fig2.gca()
# divider = make_axes_locatable(cbar_ax)
# cax = divider.append_axes("right", size="5%", pad=0.3)
# cbar = fig2.colorbar(colormap, cax=cax, orientation='vertical')
# cbar_ax.set_axis_off()
# cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)
#
#
#
#
# # Magic lattice
# fig3 = plt.figure()
# gs = GridSpec(1, 1, figure=fig3, hspace=0, wspace=0.1)
# ax = fig3.add_subplot(gs[0, 0])
# plot_info_latt(magic_latt, ax, color_map_neg, min_value=-max_value, max_value=max_value, indicate_ints=True)
# ax.set_title('Magic lattice')
#
# cbar_ax = fig3.gca()
# divider = make_axes_locatable(cbar_ax)
# cax = divider.append_axes("right", size="5%", pad=0.3)
# cbar = fig3.colorbar(colormap_neg, cax=cax, orientation='vertical')
# cbar_ax.set_axis_off()
# cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)




fig4 = plt.figure(figsize=(10, 10))
fig4.suptitle('Info lattice', fontsize=20)
gs = GridSpec(3, 5, figure=fig4, hspace=0, wspace=0.1)
ax0 = fig4.add_subplot(gs[0, 0])
ax1 = fig4.add_subplot(gs[0, 1])
ax2 = fig4.add_subplot(gs[0, 2])
ax3 = fig4.add_subplot(gs[0, 3])
ax4 = fig4.add_subplot(gs[0, 4])
# ax5 = fig4.add_subplot(gs[0, 5])
ax0_1 = fig4.add_subplot(gs[1, 0])
ax1_1 = fig4.add_subplot(gs[1, 1])
ax2_1 = fig4.add_subplot(gs[1, 2])
ax3_1 = fig4.add_subplot(gs[1, 3])
ax4_1 = fig4.add_subplot(gs[1, 4])
# ax5_1 = fig4.add_subplot(gs[1, 5])
ax0_2 = fig4.add_subplot(gs[2, 0])
ax1_2 = fig4.add_subplot(gs[2, 1])
ax2_2 = fig4.add_subplot(gs[2, 2])
ax3_2 = fig4.add_subplot(gs[2, 3])
ax4_2 = fig4.add_subplot(gs[2, 4])
# ax5_3 = fig4.add_subplot(gs[2, 5])

ax_vec = [ax0, ax1, ax2, ax3, ax4]
ax_vec_1 = [ax0_1, ax1_1, ax2_1, ax3_1, ax4_1]
ax_vec_2 = [ax0_2, ax1_2, ax2_2, ax3_2, ax4_2]

for i, ax in enumerate(ax_vec):
    plot_info_latt(info_dict[i], ax, color_map, max_value=max_value, indicate_ints=True)
for i, ax in enumerate(ax_vec_1):
    plot_magic_latt(SRE1_dict[i], ax, color_map, max_value=max_value, indicate_ints=True)
for i, ax in enumerate(ax_vec_2):
    plot_info_latt(magic_dict[i], ax, color_map_neg, min_value=-max_value, max_value=max_value, indicate_ints=True)

# cbar_ax = fig4.add_subplot(gs[0, -1])
# divider = make_axes_locatable(cbar_ax)
# cax = divider.append_axes("left", size="20%", pad=0)
# cbar = fig4.colorbar(colormap, cax=cax, orientation='vertical')
# cbar_ax.set_axis_off()
# cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)
#
# cbar_ax = fig4.add_subplot(gs[1, -1])
# divider = make_axes_locatable(cbar_ax)
# cax = divider.append_axes("left", size="20%", pad=0)
# cbar = fig4.colorbar(colormap, cax=cax, orientation='vertical')
# cbar_ax.set_axis_off()
# cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)
#
# cbar_ax = fig4.add_subplot(gs[2, -1])
# divider = make_axes_locatable(cbar_ax)
# cax = divider.append_axes("left", size="20%", pad=0)
# cbar = fig4.colorbar(colormap_neg, cax=cax, orientation='vertical')
# cbar_ax.set_axis_off()
# cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)

#
# fig5 = plt.figure(figsize=(15, 15))
# fig6 = plt.figure(figsize=(15, 15))
# fig7 = plt.figure(figsize=(15, 15))
# fig8 = plt.figure(figsize=(15, 15))
# fig9 = plt.figure(figsize=(15, 15))
# fig_list = [fig5, fig6, fig7, fig8, fig9]
# for step in prob_dist.keys():
#     fig_list[step].suptitle(f'Step {step}', fontsize=20)
#     plot_probabilities(prob_dist[step], 4, fig_list[step])


plt.show()
