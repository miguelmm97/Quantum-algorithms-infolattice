import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from scipy.interpolate import PchipInterpolator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pdf2image import convert_from_path
from PIL import Image, ImageChops
import seaborn as sns
from modules.InfoLattice import plot_info_latt
from modules.functions import *

from load import (
    load_Gamma,
    load_Lambda,
    load_information_per_scale,
    load_information_lattice
)



#%% Configuration
DATA_DIR = 'final-data-potts'
SYSTEM_SIZES = [8, 10, 12, 14] #[8, 9, 10, 11, 12]
J_VALUE = 1.0
H_VALUES = sorted({
    float(m.group('h'))
    for fname in os.listdir(DATA_DIR)
    if (m := re.match(
          rf'lowEnergyStates_N(?P<N>\d+)_J{J_VALUE:.3f}_h(?P<h>\d+\.\d{{3}})\.h5',
          fname))
    and int(m.group('N')) in SYSTEM_SIZES
})
INFO_H = [0, 0.75]



#%% Figure

# Format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']
fontsize=20

fig1 = plt.figure(figsize=(8, 8))
gs = GridSpec(4, 1, figure=fig1, wspace=0.5, hspace=0.6)
ax00 = fig1.add_subplot(gs[0, 0])
ax1 = fig1.add_subplot(gs[1:, 0])
ax0 = ax1.inset_axes([0.46, 0.32, 0.55, 0.55])
ax2 = ax1.inset_axes([0.07, 0.1, 0.28, 0.3])


# Colormap
min, max = 0, 2
palette_info = sns.color_palette("Blues", as_cmap=True)
colors_info = palette_info(np.linspace(0.1, 0.9, 100))
colors_info[0] = [1, 1, 1, 1]
colormap_info = LinearSegmentedColormap.from_list("custom_colormap", colors_info)
norm = Normalize(vmin=min, vmax=max)
colorbar_info = cm.ScalarMappable(norm=Normalize(vmin=min, vmax=max), cmap=colormap_info)
color_ints = 'royalblue'
line_ints = 2
alpha_ints = 1
color_info_per_scale1 = colors_info[75]
color_info_per_scale2 = 'crimson'
colors_scale = [color_info_per_scale1, color_info_per_scale2]
colors_L = [ colors_info[20], colors_info[60], colors_info[90], 'k']



# ---------------- Folding sketch ----------------------------------------------------------
# Load image and crop whitespace
images = convert_from_path("sketch.pdf")
images[0].save("sketch.png", "PNG")
im = Image.open("sketch.png")
bg = Image.new(im.mode, im.size, im.getpixel((0,0)))  # background color
diff = ImageChops.difference(im, bg)
bbox = diff.getbbox()
if bbox:
    im = im.crop(bbox)

ax00.imshow(im)
ax00.axis('off')
ax00.text(0, 0, '$(a)$', fontsize=fontsize)
ax00.text(-80, 330, '$n$: $0$', fontsize=fontsize)
ax00.text(200, 330, '$1$', fontsize=fontsize)
ax00.text(360, 330, '$2$', fontsize=fontsize)
ax00.text(530, 330, '$3$', fontsize=fontsize)
ax00.text(690, 330, '$4$', fontsize=fontsize)
ax00.text(850, 330, '$5$', fontsize=fontsize)
ax00.text(970, 120, 'Folding', fontsize=fontsize-5)
# ax00.text(1100, 400, '$n_{\\rm folded}$: $0$', fontsize=fontsize)
ax00.text(1280, 400, '$n\'$: $0$', fontsize=fontsize)
ax00.text(1580, 400, '$1$', fontsize=fontsize)
ax00.text(1740, 400, '$2$', fontsize=fontsize)
ax00.text(1255, -20, '$n$: $2  {\\scriptstyle \\cup}$'+'$3$', fontsize=fontsize)
ax00.text(1535, -20, '$1  {\\scriptstyle \\cup}$'+'$4$', fontsize=fontsize)
ax00.text(1700, -20, '$0  {\\scriptstyle \\cup}$'+'$5$', fontsize=fontsize)




# -----------------Fig 2(a): Example information lattice ---------------------------------------------------------------

path = os.path.join(DATA_DIR, f'lowEnergyStates_N{6}_J{J_VALUE:.3f}_h{0.0:.3f}.h5')
info_latt_load = load_information_lattice(d=0, file_path=path)[0]
info_latt = {}
L = 12

for l in range(len(info_latt_load)):
    local_info = []
    for n in range(len(info_latt_load[l])):
        if n < L - l:
            local_info.append(info_latt_load[l][n])
    info_latt[l+1] = np.array(local_info)

plot_info_latt(info_latt, ax0, colormap_info,
               indicate_ints=True,
               linewidth_ints=line_ints,
               color_ints=color_ints,
               alpha_ints=alpha_ints,
               linewidth=0.5,
               decrease_zeros=True)
ax0.set_xlabel('$n$', fontsize=fontsize, labelpad=-15)
ax0.set_ylabel('$\ell$', fontsize=fontsize, labelpad=-20)
# ax0.set_xlim(-0.1, 1.1)
# ax0.set_ylim(0.0, 1.0)
ax0.set_xticks(ticks=[0.025, 0.975], labels=['0', f'{11}'])
ax0.set_yticks(ticks=[0.025, 0.97], labels=['0', f'{11}'])
ax0.tick_params(which='major', width=0.75, labelsize=fontsize)
ax0.tick_params(which='major', length=6, labelsize=fontsize)
ax0.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax0.tick_params(which='minor', length=3, labelsize=fontsize)
ax0.axis('on')

# Colorbar
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="6%", pad=0.1)
cbar = fig1.colorbar(colorbar_info, cax=cax, orientation='vertical', ticks=[0, 1, 2])
# cbar_ax.set_axis_off()
cbar.set_label(label='$i^{\ell}_n$', labelpad=-3, fontsize=20, rotation='horizontal')
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
cbar.ax.set_yticklabels(['0', '1', '2'])
label = cbar.ax.yaxis.label
label.set_position((-0.3, 0.8))


# -----------------Fig 2(c): Gamma/ Lambda vs h ------------------------------------------------------------------------

for i, N in enumerate(SYSTEM_SIZES):
    gamma_vals, lambda_vals = [], []
    for h in H_VALUES:
        path = os.path.join(
            DATA_DIR,
            f'lowEnergyStates_N{N}_J{J_VALUE:.3f}_h{h:.3f}.h5'
        )
        if os.path.exists(path):
            gamma, _ = load_Gamma(d=0, file_path=path)
            lam,   _ = load_Lambda(d=0, file_path=path)
            gamma_vals.append(np.real(gamma))
            lambda_vals.append(np.real(lam))
        else:
            gamma_vals.append(np.nan)
            lambda_vals.append(np.nan)

    ax1.plot(H_VALUES, gamma_vals,
             marker='o',
             markersize=7,
             markerfacecolor='None',
             linestyle='solid',
             linewidth=1.5,
             color=colors_L[i],
             label=f'${2 * N}$',
             alpha=1)
    ax1.plot(H_VALUES, lambda_vals,
             marker='^',
             markerfacecolor='None',
             markersize=7,
             linestyle='dashed',
             linewidth=1.5,
             color=colors_L[i],
             alpha=1)

ax1.plot(np.linspace(-0.1, 0.8, 10), np.ones((10, )) * np.log2(3), linestyle='dashed', color='grey', alpha=0.5)
ax1.set_xlabel('$h$', fontsize=fontsize)
ax1.set_xlim(0, 0.8)
ax1.set_ylim(0-0.015, 1.75)
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)
ax1.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax1.tick_params(which='minor', length=3, labelsize=fontsize)
ax1.set_xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8])
ax1.set_yticks(ticks=[0, 0.5, 1, 1.5])
xminor_ticks = [0.1, 0.3, 0.5, 0.7]
yminor_ticks = [0.25, 0.75, 1.25, 1.75]
ax1.yaxis.set_minor_locator(plt.FixedLocator(yminor_ticks))
ax1.xaxis.set_minor_locator(plt.FixedLocator(xminor_ticks))
label = ax1.yaxis.get_label()
x, y = label.get_position()
label.set_position((x, y - 0.1))

ax1.text(0.01, np.log2(3) + 0.05, '$(b)$', fontsize=fontsize)
ax1.text(0.7, np.log2(3) + 0.05, '$\\log_2 (3)$', fontsize=fontsize, color='grey')
ax1.text(0.1385, 1.3, '$\\underline{L}$', fontsize=fontsize)
leg1 = ax1.legend(loc='center left',
           ncol=1,
           handlelength=0.7,
           columnspacing=0.4,
           labelspacing=0.05,
           frameon=False,
           fontsize=fontsize,
           bbox_to_anchor=(0.06, 0.6))

for line in leg1.get_lines():
    line.set_marker('None')
    line.set_linewidth(2)

ax1.text(0.65, 0.3, '$\\Gamma$', fontsize=fontsize)
ax1.text(0.65, 0.2, '$\\Gamma_{\\rm folded}$', fontsize=fontsize)
ax1.plot(0.63, 0.23, marker='^', color='k', markerfacecolor='None', markersize=7, alpha=1)
ax1.plot(0.63, 0.33, marker='o', color='k', markersize=7, markerfacecolor='None', alpha=1)


# ---------------------------Fig 2(d): Info per scale ------------------------------------------------------------------
scales = None
L = 12
for j, h in enumerate(INFO_H):
    path = os.path.join(
        DATA_DIR,
        f'lowEnergyStates_N6_J{J_VALUE:.3f}_h{h:.3f}.h5'
    )
    if not os.path.exists(path):
        continue
    data, attrs = load_information_per_scale(d=0, file_path=path)
    vals = np.vstack([data[field] for field in data.dtype.names]).T
    y = vals[0, :-1]
    scales = np.arange(len(y)) if scales is None else scales
    interp_func = PchipInterpolator(scales, y)
    interp_scales = np.arange(scales[0]-0.2, scales[-1] + 0.1, 0.1)
    interp_average = interp_func(interp_scales)

    if j==0:
        gamma = np.sum(y[int(0.5 * L):])
        zorder=10
    else:
        zorder=0

    ax2.plot(interp_scales, interp_average,
             marker='None',
             linestyle='solid',
             color=colors_scale[j],
             label=fr'${h:.2f}$',
             zorder=zorder)
    ax2.fill_between(interp_scales, interp_average, -0.2,
             color=colors_scale[j], alpha=0.2, zorder=zorder)
    ax2.plot(scales, y,
             marker='o',
             markersize=4,
             linestyle='None',
             color=colors_scale[j],
             zorder=zorder)



ax2.set_xlabel('$\ell$', fontsize=fontsize, labelpad=-17)
ax2.set_ylabel('$I^\ell$', fontsize=fontsize, rotation='horizontal', labelpad=-13)
ax2.set_yticks(ticks=[0, 10], labels=[0, 10])
ax2.set_xticks(ticks=[0, 5, 11], labels=[0, 5, 11])
label = ax2.yaxis.get_label()
x, y = label.get_position()
label.set_position((x, y - 0.1))
label = ax2.xaxis.get_label()
x, y = label.get_position()
label.set_position((x+0.2, y))
ax2.set_xlim(-0.2, 11.1)
ax2.set_ylim(-0.2, 10.1)
ax2.tick_params(which='major', width=0.75, labelsize=fontsize)
ax2.tick_params(which='major', length=6, labelsize=fontsize)
ax2.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax2.tick_params(which='minor', length=3, labelsize=fontsize)
ax2.text(1, 7.3, '$\\underline{h}$', fontsize=fontsize-3)
yminor_ticks = [6]
ax2.yaxis.set_minor_locator(plt.FixedLocator(yminor_ticks))
ax2.legend(loc='upper center',
           ncol=2,
           handlelength=0.5,
           columnspacing=0.4,
           labelspacing=0.1,
           frameon=False,
           fontsize=fontsize-5,
           handletextpad=0.3,
           bbox_to_anchor=(0.61, 1))
ax2.text(4.8, 1.5, f'$\\Gamma={gamma :.3f}$', fontsize=fontsize-5, color=colors_scale[0])

# Save figure with tight white boundary
plt.savefig("fig-2.pdf", bbox_inches='tight', pad_inches=0.1)
plt.show()

