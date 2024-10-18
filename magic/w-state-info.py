#%% Imports

# Built-in modules
import math
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
from InfoLattice import calc_info, plot_info_latt, calc_info_per_scale
from functions import random_wn_state

#%%

n = 10
k = 10
psi = random_wn_state(n, k)
info = calc_info(psi)



# Normalising the colormap
max_value = 2
color_map = plt.get_cmap("Oranges")
colors = color_map(np.linspace(0, 1, 20))
colors[0] = [1, 1, 1, 1]
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)
colormap = cm.ScalarMappable(norm=Normalize(vmin=0., vmax=max_value), cmap=color_map)


fig1 = plt.figure()
gs = GridSpec(1, 1, figure=fig1, hspace=0, wspace=0.1)
ax = fig1.add_subplot(gs[0, 0])
plot_info_latt(info, ax, color_map, max_value=max_value, indicate_ints=True)
plt.show()

# cbar_ax = fig1.add_subplot(gs[1, -1])
# divider = make_axes_locatable(cbar_ax)
# cax = divider.append_axes("left", size="20%", pad=0)
# cbar = fig1.colorbar(colormap, cax=cax, orientation='vertical')
# cbar_ax.set_axis_off()
# cbar.set_label(label='$i_n^l$', labelpad=10, fontsize=20)