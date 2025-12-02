
#%% Imports

# Built-in modules
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pdf2image import convert_from_path
from PIL import Image, ImageChops
import seaborn as sns

# Information lattice and functions
from InfoLattice import plot_info_latt
from functions import *


#%% Load data
file_list = ["data-fig2.h5"]
data_dict = load_my_data(file_list, "../data")

# Main figure
system_sizes = data_dict[file_list[0]]['main_figure']['system_size']
J_values = data_dict[file_list[0]]['main_figure']['J_values']
h_values = data_dict[file_list[0]]['main_figure']['h_values']
gamma_values = data_dict[file_list[0]]['main_figure']['gamma_values']
gamma_folded_values = data_dict[file_list[0]]['main_figure']['gamma_folded_values']

# Left inset
h_value_left = data_dict[file_list[0]]['left_inset']['h_values']
J_values_left = data_dict[file_list[0]]['left_inset']['J_values']
L_value_left = data_dict[file_list[0]]['left_inset']['L_value']
interp_scales_0 = data_dict[file_list[0]]['left_inset']['interp_scales_0']
interp_avg_0 = data_dict[file_list[0]]['left_inset']['interp_avg_0']
info_scale_0 = data_dict[file_list[0]]['left_inset']['info_scale_0']
interp_scales_075 = data_dict[file_list[0]]['left_inset']['interp_scales_075']
interp_avg_075 = data_dict[file_list[0]]['left_inset']['interp_avg_075']
info_scale_075 = data_dict[file_list[0]]['left_inset']['info_scale_075']

# Right inset
h_values_right = data_dict[file_list[0]]['right_inset']['h_value']
L_value_right = data_dict[file_list[0]]['right_inset']['L_value']
info_latt_load = data_dict[file_list[0]]['right_inset']['info_lattice']



#%% Figures

# Format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']
fontsize=20

# Figure grid
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
colorbar_info = cm.ScalarMappable(norm=Normalize(vmin=min, vmax=max), cmap=colormap_info)
color_ints = 'royalblue'
line_ints = 2
alpha_ints = 1
color_info_per_scale1 = colors_info[75]
color_info_per_scale2 = 'crimson'
colors_scale = [color_info_per_scale1, color_info_per_scale2]
colors_L = [ colors_info[20], colors_info[60], colors_info[90], 'k']

min, max = 0, 1
palette_info_lattice = sns.color_palette("Blues", as_cmap=True)
colors_info_lattice = palette_info_lattice(np.linspace(0.1, 0.9, 100))
colors_info_lattice[0] = [1, 1, 1, 1]
colormap_info_lattice = LinearSegmentedColormap.from_list("custom_colormap", colors_info_lattice)
colorbar_info_lattice = cm.ScalarMappable(norm=Normalize(vmin=min, vmax=max), cmap=colormap_info_lattice)



# ---------------- Folding sketch --------------------------------------------------------------------------------------
# Load image and crop whitespace
images = convert_from_path("sketch.pdf")
images[0].save("sketch.png", "PNG")
im = Image.open("./sketch.png")
bg = Image.new(im.mode, im.size, im.getpixel((0,0)))  # background color
diff = ImageChops.difference(im, bg)
bbox = diff.getbbox()
if bbox:
    im = im.crop(bbox)
im = im.resize((im.width * 3, im.height * 3), Image.LANCZOS)
im = np.asarray(im)

ax00.imshow(im, interpolation='none')
ax00.axis('off')
ax00.text(0, 0, '$(a)$', fontsize=fontsize)
ax00.text(-5 * 3, 330 * 3, '$n$: $0$', fontsize=fontsize)
ax00.text(285 * 3, 330 * 3, '$1$', fontsize=fontsize)
ax00.text(445 * 3, 330 * 3, '$2$', fontsize=fontsize)
ax00.text(610 * 3, 330 * 3, '$3$', fontsize=fontsize)
ax00.text(770 * 3, 330 * 3, '$4$', fontsize=fontsize)
ax00.text(940 * 3, 330 * 3, '$5$', fontsize=fontsize)
ax00.text(1035 * 3, 120 * 3, 'Folding', fontsize=fontsize-5)
ax00.text(1260 * 3, 410 * 3, '$n\'$:', fontsize=fontsize)
ax00.text(1390 * 3, 410 * 3, '$0$', fontsize=fontsize)
ax00.text(1575 * 3, 410 * 3, '$1$', fontsize=fontsize)
ax00.text(1740 * 3, 410 * 3, '$2$', fontsize=fontsize)
ax00.text(1260 * 3, -20 * 3, '$n$:', fontsize=fontsize)
ax00.text(1365 * 3, -20 * 3, '$0  {\\scriptstyle \\cup}$'+'$5$', fontsize=fontsize)
ax00.text(1530 * 3, -20 * 3, '$1  {\\scriptstyle \\cup}$'+'$4$', fontsize=fontsize)
ax00.text(1705 * 3, -20 * 3, '$2  {\\scriptstyle \\cup}$'+'$3$', fontsize=fontsize)




# -----------------Fig 2(a): Example information lattice ---------------------------------------------------------------

# Load info lattice to a dictionary
info_latt = {}
for l in range(len(info_latt_load)):
    local_info = []
    for n in range(len(info_latt_load[l])):
        if n < L_value_right - l:
            local_info.append(info_latt_load[l][n])
    info_latt[l+1] = np.array(local_info)

plot_info_latt(info_latt, ax0, colormap_info_lattice,
               indicate_ints=True,
               linewidth_ints=line_ints,
               color_ints=color_ints,
               alpha_ints=alpha_ints,
               linewidth=0.15,
               max_value=1)
ax0.set_xlabel('$n$', fontsize=fontsize, labelpad=-15)
ax0.set_ylabel('$\ell$', fontsize=fontsize, labelpad=-20)
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
cbar = fig1.colorbar(colorbar_info_lattice, cax=cax, orientation='vertical', ticks=[0, 1])
cbar.set_label(label='$i^{\ell}_n$', labelpad=-3, fontsize=20, rotation='horizontal')
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
cbar.ax.set_yticklabels(['0', '1'])
label = cbar.ax.yaxis.label
label.set_position((-0.3, 0.6))
ax0.text(0.05, 0.85, '$h=0$', fontsize=fontsize)


# -----------------Fig 2(c): Gamma/ Gamma folded vs h ------------------------------------------------------------------------

for i, N in enumerate(system_sizes):
    ax1.plot(h_values, gamma_values[i, :],
             marker='o',
             markersize=7,
             markerfacecolor='None',
             linestyle='solid',
             linewidth=1.5,
             color=colors_L[i],
             label=f'${2 * N}$',
             alpha=1)
    ax1.plot(h_values, gamma_folded_values[i, :],
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
scales = np.arange(0, L_value_left)

ax2.plot(interp_scales_0, interp_avg_0,
         marker='None',
         linestyle='solid',
         color=colors_scale[0],
         label=fr'${h_value_left[0]:.2f}$',
         zorder=10)
ax2.fill_between(interp_scales_0, interp_avg_0, -0.2,
         color=colors_scale[0],
         alpha=0.2,
         zorder=10)
ax2.plot(scales, info_scale_0,
         marker='o',
         markersize=4,
         linestyle='None',
         color=colors_scale[0],
         zorder=10)
ax2.plot(interp_scales_075, interp_avg_075,
         marker='None',
         linestyle='solid',
         color=colors_scale[1],
         label=fr'${h_value_left[1]:.2f}$',
         zorder=0)
ax2.fill_between(interp_scales_075, interp_avg_075, -0.2,
         color=colors_scale[1],
         alpha=0.2,
         zorder=0)
ax2.plot(scales, info_scale_075,
         marker='o',
         markersize=4,
         linestyle='None',
         color=colors_scale[1],
         zorder=0)

gamma = np.sum(info_scale_0[int(0.5 * L_value_left):])

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
ax2.text(1.1, 7.6, '$\\underline{h}$', fontsize=fontsize-3)
yminor_ticks = [6]
ax2.yaxis.set_minor_locator(plt.FixedLocator(yminor_ticks))
ax2.legend(loc='upper center',
           ncol=2,
           handlelength=0.5,
           columnspacing=0.5,
           labelspacing=0.1,
           frameon=False,
           fontsize=fontsize-5,
           handletextpad=0.3,
           bbox_to_anchor=(0.61, 1))
ax2.text(4.8, 1.5, f'$\\Gamma={gamma :.3f}$', fontsize=fontsize-5, color=colors_scale[0])

# Save figure with tight white boundary
plt.savefig("fig-2.pdf", bbox_inches='tight', pad_inches=0.1)
plt.show()

