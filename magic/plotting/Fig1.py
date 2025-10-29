#%% Imports

# Built-in modules
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import matplotlib.patches as patches
from scipy.interpolate import splprep, splev
from matplotlib.path import Path

# Managing data
import os
import sys
import h5py
from datetime import date

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Information lattice and functions
from modules.InfoLattice import calc_info, plot_info_latt, calc_info_per_scale
from modules.functions import *


file_list = ['Exp32.h5']
data_dict = load_my_data(file_list, '..')
seedlist1 = data_dict[file_list[0]]['Simulation']['seed_stab']
seedlist2 = data_dict[file_list[0]]['Simulation']['seed_srm']

#%% Parameters

# Example 1: Néel state
psi0 = Statevector.from_label('0101')
info_latt_0 = calc_info(psi0.data)

# Example 2: Néel + Bell pair
psi1 = (1 / np.sqrt(2)) * (Statevector.from_label('0101') + Statevector.from_label('0011'))
info_latt_1 = calc_info(psi1.data)

# Example 3: 4-qubit GHZ state
psi_GHZ =  Statevector.from_label('0000')
GHZ_circuit = QuantumCircuit(4)
GHZ_circuit.h(0)
for i in range(1, 4):
    GHZ_circuit.cx(i - 1, i)
psi_GHZ = psi_GHZ.evolve(GHZ_circuit)
info_latt_2 = calc_info(psi_GHZ.data)

# Example 4: Random stab from the GHZ state
n_qubits = 10
psi_GHZ =  Statevector.from_label('0' * n_qubits)
GHZ_circuit = QuantumCircuit(n_qubits)
GHZ_circuit.h(0)
for i in range(1, n_qubits):
    GHZ_circuit.cx(i - 1, i)
psi_GHZ = psi_GHZ.evolve(GHZ_circuit)
# seedlist1 = np.random.randint(0, 1000000, size=(2, ))
clifford = random_clifford_circuit(num_qubits=n_qubits, depth=2, seed=seedlist1)
psi_stab = psi_GHZ.evolve(clifford)
info_latt = calc_info(psi_stab.data)
info_per_scale_2 = calc_info_per_scale(info_latt, bc='open')
scales = np.arange(0, 10)
interp_func = PchipInterpolator(scales, info_per_scale_2 ) #/ np.arange(n_qubits + 1, 1, -1))
interp_scales_2 = np.arange(scales[0] - 0.2, scales[-1] + 0.1, 0.1)
interp_average_2 = interp_func(interp_scales_2)


# Example 5: Short-range nonstabilizer state
n_qubits = 10
psi_srm_label = '0' * n_qubits
psi_srm = Statevector.from_label(psi_srm_label)
Nlayers, Nblocks = 10, 3
T_per_block = 5
min_block, max_block = 0, 20
qubits = list(range(n_qubits))
clifford_sequence = QuantumCircuit(n_qubits)
magic_circuit = QuantumCircuit(n_qubits)
info_interval = int(Nlayers/ Nblocks)
# seedlist2 = np.random.randint(0, 1000000, size=(Nlayers, ))

for i in range(Nlayers):
    # Clifford part
    layer = QuantumCircuit(n_qubits)
    clifford = random_clifford_circuit(num_qubits=n_qubits, depth=1, seed=seedlist2[i])
    layer.compose(clifford, inplace=True)

    # Application of T gates
    rng = np.random.default_rng(seedlist2[i])
    rng.shuffle(qubits)
    operands = qubits[-T_per_block:]
    if (i % info_interval) == 0 and min_block < (i // info_interval) < max_block:
        for qubit in operands:
            layer.t(qubit)
    magic_circuit.compose(layer, inplace=True)

    # Circuit evolution
    psi_srm = psi_srm.evolve(layer)

info_latt_srm = calc_info(psi_srm.data)
info_per_scale_srm = calc_info_per_scale(info_latt_srm, bc='open')
scales = np.arange(0, 10)
interp_func = PchipInterpolator(scales, info_per_scale_srm)
interp_scales_srm = np.arange(scales[0] - 0.2, scales[-1] + 0.1, 0.1)
interp_average_srm = interp_func(interp_scales_srm)


# data_dir = '..'
# file_list = os.listdir(data_dir)
# expID = get_fileID(file_list, common_name='Exp')
# filename = '{}{}{}'.format('Exp', expID, '.h5')
# filepath = os.path.join(data_dir, filename)

# with h5py.File(filepath, 'w') as f:
#     simulation = f.create_group('Simulation')
#     store_my_data(simulation, 'seed_stab', seedlist1)
#     store_my_data(simulation, 'seed_srm', seedlist2)





#%% Figures
# Format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']
fontsize=20

# Colormap
min, max = 0, 2
palette_info = sns.color_palette("Blues", as_cmap=True)
colors_info = palette_info(np.linspace(0.1, 0.9, 100))
colors_info[0] = [1, 1, 1, 1]
colormap_info = LinearSegmentedColormap.from_list("custom_colormap", colors_info)
norm = Normalize(vmin=min, vmax=max)
colorbar_info = cm.ScalarMappable(norm=Normalize(vmin=min, vmax=max), cmap=colormap_info)
color_ints = 'k'
line_ints = 2
line_sites = 0.5
alpha_ints = 1
color_info_per_scale1 = colors_info[75]
color_info_per_scale2 = colors_info[35]

# Figure grid
fig1 = plt.figure(figsize=(12, 6))
gs = GridSpec(4, 3, figure=fig1, wspace=-0.1, hspace=0.65)
ax0 = fig1.add_subplot(gs[0:2, 0])
ax1 = fig1.add_subplot(gs[0:2, 1])
ax2 = fig1.add_subplot(gs[0:2, 2])
ax3 = fig1.add_subplot(gs[2:, 0])
ax4 = fig1.add_subplot(gs[2:, 1])
ax5 = fig1.add_subplot(gs[2:, 2])
inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=ax5, wspace=0., hspace=0.3)
ax5_1 = fig1.add_subplot(inner_gs[0, 0])
ax5_2 = fig1.add_subplot(inner_gs[1, 0])
ax5.axis('off')
ax5.set_xlim(-0.1, 1.1)
ax5.set_ylim(0.0, 1.0)

# Figure labels
pos0 = ax0.get_position()
pos1 = ax1.get_position()
pos2 = ax2.get_position()
pos3 = ax3.get_position()
pos4 = ax4.get_position()
pos5_1 = ax5_1.get_position()
pos5_2 = ax5_2.get_position()
fig1.text(pos0.x0 + 0.058,   pos0.y0 + 0.3, '$(a)$', fontsize=fontsize-3, ha="center")
fig1.text(pos1.x0 + 0.058,   pos1.y0 + 0.3, '$(b)$', fontsize=fontsize-3, ha="center")
fig1.text(pos2.x0 + 0.058,   pos2.y0 + 0.3, '$(c)$', fontsize=fontsize-3, ha="center")
fig1.text(pos3.x0 + 0.058,   pos3.y0 + 0.3, '$(d)$', fontsize=fontsize-3, ha="center")
fig1.text(pos4.x0 + 0.058,   pos4.y0 + 0.3, '$(e)$', fontsize=fontsize-3, ha="center")
fig1.text(pos5_1.x0 + 0.054, pos5_1.y0 + 0.11, '$(f)$', fontsize=fontsize-3, ha="center")
fig1.text(pos5_2.x0 + 0.054, pos5_2.y0 + 0.1, '$(g)$', fontsize=fontsize-3, ha="center")


# -------------Fig 1(a): Néel-------------------------------------------------------------------------------------------
plot_info_latt(info_latt_0, ax0, colormap_info,
               indicate_ints=True,
               linewidth_ints=line_ints,
               color_ints=color_ints,
               alpha_ints=alpha_ints,
               linewidth=line_sites)
ax0.set_xlabel('$n$', fontsize=fontsize, labelpad=-15)
ax0.set_ylabel('$\ell$', fontsize=fontsize, labelpad=-15)
ax0.set_xlim(-0.1, 1.1)
ax0.set_ylim(0.0, 1.0)
ax0.set_xticks(ticks=[0.12, 0.88], labels=['0', f'{4 - 1}'])
ax0.set_yticks(ticks=[0.12, 0.88], labels=['0', f'{4 - 1}'])
ax0.tick_params(which='major', width=0.75, labelsize=fontsize)
ax0.tick_params(which='major', length=6, labelsize=fontsize)
ax0.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax0.tick_params(which='minor', length=3, labelsize=fontsize)
ax0.text(0.75, 0.85, '$\\vert$Néel$\\rangle$', fontsize=fontsize-3)
ax0.axis('on')


# --------------Example 1(b): Bell pair---------------------------------------------------------------------------------
plot_info_latt(info_latt_1, ax1, colormap_info,
               indicate_ints=True,
               linewidth_ints=line_ints,
               color_ints=color_ints,
               alpha_ints=alpha_ints,
               linewidth=line_sites)
ax1.set_xlabel('$n$', fontsize=fontsize, labelpad=-15)
ax1.set_ylabel('$\ell$', fontsize=fontsize, labelpad=-15)
ax1.set_xlim(-0.1, 1.1)
ax1.set_ylim(0.0, 1.0)
ax1.set_xticks(ticks=[0.12, 0.88], labels=['0', f'{4 - 1}'])
ax1.set_yticks(ticks=[0.12, 0.88], labels=['0', f'{4 - 1}'])
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)
ax1.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax1.tick_params(which='minor', length=3, labelsize=fontsize)
ax1.text(0.75, 0.85, '$\\vert$Bell$\\rangle$', fontsize=fontsize-3)
ax1.text(0.42, 0.19, '$\mathcal{C}_{1.5}^1$', fontsize=fontsize-5, color=color_info_per_scale1, alpha=1)
ax1.axis('on')

# Triangle denoting subsystem C_1.5^1
def rounded_triangle(points, radius=0.05, color='k', alpha=1):
    pts = np.array(points)
    n = len(pts)
    path_data = []
    for i in range(n):
        p_prev = pts[i - 1]
        p_curr = pts[i]
        p_next = pts[(i + 1) % n]
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue
        v1 /= n1
        v2 /= n2

        start = p_curr + v1 * radius
        end   = p_curr + v2 * radius

        if i == 0:
            path_data.append((Path.MOVETO, start))
        else:
            path_data.append((Path.LINETO, start))
        path_data.append((Path.CURVE3, p_curr))
        path_data.append((Path.CURVE3, end))

    path_data.append((Path.LINETO, path_data[0][1]))
    path_data.append((Path.CLOSEPOLY, path_data[0][1]))
    codes, verts = zip(*path_data)
    return patches.PathPatch(Path(verts, codes), facecolor='none', edgecolor=color, lw=2, alpha=alpha)
tri_patch = rounded_triangle([[0.2, 0.02], [0.8, 0.02], [0.5, 0.59]],
                             radius=0.25,
                             color=color_info_per_scale1,
                             alpha=0.5)
ax1.add_patch(tri_patch)


# ---------------Fig 1(c): GHZ 4 qubits---------------------------------------------------------------------------------
plot_info_latt(info_latt_2, ax2, colormap_info,
               indicate_ints=True,
               linewidth_ints=line_ints,
               color_ints=color_ints,
               alpha_ints=alpha_ints,
               linewidth=line_sites)
ax2.set_xlabel('$n$', fontsize=fontsize, labelpad=-15)
ax2.set_ylabel('$\ell$', fontsize=fontsize, labelpad=-15)
ax2.set_xlim(-0.1, 1.1)
ax2.set_ylim(0.0, 1.0)
ax2.set_xticks(ticks=[0.12, 0.88], labels=['0', f'{4 - 1}'])
ax2.set_yticks(ticks=[0.12, 0.88], labels=['0', f'{4 - 1}'])
ax2.text(0.72, 0.85, '$\\vert$GHZ$\\rangle$', fontsize=fontsize-3)
ax2.tick_params(which='major', width=0.75, labelsize=fontsize)
ax2.tick_params(which='major', length=6, labelsize=fontsize)
ax2.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax2.tick_params(which='minor', length=3, labelsize=fontsize)
ax2.axis('on')


# Stabilizer group to the side
ax2.text(1.28, 0.92, '$\\mathcal{G}_{\\rm GHZ}$ ', fontsize=fontsize-2)
ax2.text(1.15, 0.77, '$\\underline{\ell = 3}$', fontsize=fontsize-5)
ax2.text(1.45, 0.77, '$\\underline{\ell = 2}$', fontsize=fontsize-5)
ax2.text(1.45, 0.5, '$\\underline{\ell = 1}$', fontsize=fontsize - 5)
ax2.text(1.45, 0.15, '$\\underline{\ell = 0}$', fontsize=fontsize - 5)
text_l3 = '\n'.join(['$XXXX$', '$XYYX$', '$YXYX$', '$YYXX$', '$XXYY$', '$XYXY$', '$YXXY$', '$YYYY$', '$ZZZZ$', '$ZIIZ$'])
text_l2 = '\n'.join(['$ZIZI$', '$IZIZ$'])
text_l1 = '\n'.join(['$ZZII$', '$IZZI$', '$IIZZ$'])
ax2.text(1.15, 0.05, text_l3, fontsize=fontsize-10, family='monospace')
ax2.text(1.47, 0.6, text_l2, fontsize=fontsize-10, family='monospace')
ax2.text(1.47, 0.27, text_l1, fontsize=fontsize-10, family='monospace')
ax2.text(1.55, 0.05, '$\\emptyset$', fontsize=fontsize-7, family='monospace')


# ------------------Fig1(d) Random stab---------------------------------------------------------------------------------
plot_info_latt(info_latt, ax3, colormap_info,
               indicate_ints=True,
               linewidth_ints=line_ints,
               color_ints=color_ints,
               alpha_ints=alpha_ints,
               linewidth=line_sites)
ax3.set_xlabel('$n$', fontsize=fontsize, labelpad=-15)
ax3.set_ylabel('$\ell$', fontsize=fontsize, labelpad=-15)
ax3.set_xlim(-0.1, 1.1)
ax3.set_ylim(0.0, 1.0)
ax3.set_xticks(ticks=[0.05, 0.95], labels=['0', f'{9}'])
ax3.set_yticks(ticks=[0.05, 0.95], labels=['0', f'{9}'])
ax3.text(0.62, 0.85, '$\\mathcal{U}_{\\mathcal{C}} \\vert$GHZ$\\rangle$', fontsize=fontsize-3)
ax3.tick_params(which='major', width=0.75, labelsize=fontsize)
ax3.tick_params(which='major', length=6, labelsize=fontsize)
ax3.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax3.tick_params(which='minor', length=3, labelsize=fontsize)
ax3.axis('on')


# -------------------Fig 1(e): Short-range magic------------------------------------------------------------------------
plot_info_latt(info_latt_srm, ax4, colormap_info,
               indicate_ints=True,
               linewidth_ints=line_ints,
               color_ints=color_ints,
               alpha_ints=alpha_ints,
               linewidth=line_sites)
ax4.set_xlabel('$n$', fontsize=fontsize, labelpad=-15)
ax4.set_ylabel('$\ell$', fontsize=fontsize, labelpad=-15)
ax4.set_xlim(-0.1, 1.1)
ax4.set_ylim(0.0, 1.0)
ax4.set_xticks(ticks=[0.05, 0.95], labels=['0', f'{9}'])
ax4.set_yticks(ticks=[0.05, 0.95], labels=['0', f'{9}'])
ax4.text(0.62, 0.85, '$\\mathcal{U}_{\\mathcal{C}T} \\vert 0\\rangle ^{\\otimes 10}$', fontsize=fontsize-3)
ax4.tick_params(which='major', width=0.75, labelsize=fontsize)
ax4.tick_params(which='major', length=6, labelsize=fontsize)
ax4.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax4.tick_params(which='minor', length=3, labelsize=fontsize)
ax4.axis('on')


# -------------------Fig 1(g): Info per scale SRM---------------------------------------------------------------
ax5_1.plot(interp_scales_srm, interp_average_srm,
           marker='None',
           color=color_info_per_scale1)
ax5_1.plot(scales, info_per_scale_srm,
         marker='o',
         markersize=4,
         linestyle='none',
         color=color_info_per_scale1)
ax5_1.fill_between(interp_scales_srm, interp_average_srm, -0.2,
                   color=color_info_per_scale1,
                   alpha=0.2)
ax5_1.set_xlim(-0.2, n_qubits - 1+0.2)
ax5_1.set_ylim(-0.2, 7+0.2)
ax5_1.set_xlabel('$\ell$', fontsize=fontsize, labelpad=-20)
ax5_1.set_ylabel('$ I^\ell$', fontsize=fontsize, rotation='horizontal', labelpad=-5)
ax5_1.set_xticks(ticks=[0, 5, 9], labels=['$0$', '$5$', '$9$'])
ax5_1.set_yticks(ticks=[0, 7])
ax5_1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax5_1.tick_params(which='major', length=6, labelsize=fontsize)
ax5_1.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax5_1.tick_params(which='minor', length=3, labelsize=fontsize)
yminor_ticks = [3.5]
ax5_1.yaxis.set_minor_locator(plt.FixedLocator(yminor_ticks))
label = ax5_1.xaxis.get_label()
x, y = label.get_position()
label.set_position((x + 0.3, y))
label = ax5_1.yaxis.get_label()
x, y = label.get_position()
label.set_position((x, y - 0.2))
ax5_1.text(7, 1, '$\\Gamma=0$', fontsize=fontsize-3)
ax5_1.text(2.4, 2.1, '$\\Omega=10$', fontsize=fontsize-3)
ax5_1.text(4, 5,  '$\\mathcal{U}_{\\mathcal{C}T} \\vert 0\\rangle ^{\\otimes 10}$', fontsize=fontsize-3)
ax5_1.set_position([pos2.x0 + 0.035, pos4.y0, pos2.width - 0.071, pos4.height * 0.4])




# -------------------Fig 1(f): Info per scale stab---------------------------------------------------------------
ax5_2.plot(interp_scales_2, interp_average_2,
           marker='None',
           color=color_info_per_scale1)
ax5_2.plot(scales, info_per_scale_2,
         marker='o',
         markersize=4,
         linestyle='None',
         color=color_info_per_scale1)
ax5_2.fill_between(interp_scales_2, interp_average_2, -0.2,
                   color=color_info_per_scale1,
                   alpha=0.2)
ax5_2.set_xlim(-0.2, n_qubits - 1+0.2)
ax5_2.set_ylim(-0.2, 7+0.2)
ax5_2.set_ylabel('$ I^\ell$', fontsize=fontsize, rotation='horizontal', labelpad=-5)
ax5_2.tick_params(which='major', width=0.75, labelsize=fontsize)
ax5_2.tick_params(which='major', length=6, labelsize=fontsize)
ax5_2.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax5_2.tick_params(which='minor', length=3, labelsize=fontsize)
ax5_2.set_xticks(ticks=[0, (n_qubits / 2) , n_qubits - 1], labels=[])
ax5_2.set_yticks(ticks=[0, 7])
yminor_ticks = [3.5]
ax5_2.yaxis.set_minor_locator(plt.FixedLocator(yminor_ticks))
ax5_2.tick_params(which='major', width=0.75, labelsize=fontsize)
label = ax5_2.yaxis.get_label()
x, y = label.get_position()
label.set_position((x, y - 0.2))
ax5_2.text(7, 2, '$\\Gamma=1$', fontsize=fontsize-3)
ax5_2.text(2, 2, '$\\Omega=9$', fontsize=fontsize-3)
ax5_2.text(4, 5,  '$\\mathcal{U}_{\\mathcal{C}} \\vert$GHZ$\\rangle$', fontsize=fontsize-3)
pos5 = ax5_1.get_position()
ax5_2.set_position([pos5.x0, pos5.y0 + 0.205, pos5.width, pos5.height])



# Colorbar
cbar_ax = fig1.add_subplot(gs[2:, 2])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("right", size="3%", pad=-5)
cbar = fig1.colorbar(colorbar_info, cax=cax, orientation='vertical', ticks=[0, 1, 2])
cbar_ax.set_axis_off()
cbar.set_label(label='$i^{\ell}_n$', labelpad=15, fontsize=20, rotation='horizontal')
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
cbar.ax.set_yticklabels(['0', '1', '2'])




fig1.savefig('fig-1.pdf', format='pdf')
plt.show()

