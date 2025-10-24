import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from load import (
    load_Gamma,
    load_Lambda,
    load_information_per_scale
)

# ─── APS / PRB FONT CONFIGURATION ─────────────────────────────────────────────
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{txfonts}",
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.dpi": 300
})

# Single-column width for PRB in inches
col_width = 3.4
# Line width
lw = 1.2

# Ensure output folder exists
os.makedirs("plots", exist_ok=True)

# CONFIGURATION
DATA_DIR = 'results_Potts'
SYSTEM_SIZES = [10, 11, 12, 13] #[8, 9, 10, 11, 12]
J_VALUE = 1.0
H_VALUES = sorted({
    float(m.group('h'))
    for fname in os.listdir(DATA_DIR)
    if (m := re.match(
          rf'lowEnergyStates_N(?P<N>\d+)_J{J_VALUE:.3f}_h(?P<h>\d+\.\d{{3}})\.h5',
          fname))
    and int(m.group('N')) in SYSTEM_SIZES
})
INFO_H = [0., 0.25, 0.5, 0.75]

cmap_sys = plt.get_cmap('plasma', len(SYSTEM_SIZES) + 1)
cmap_info = plt.get_cmap('plasma', len(INFO_H) + 1)
markers = ['o', 's', '^', 'v', 'p', '*']
lines = ['-', '--', '-.', ':']

# ─── 1) Gamma & Lambda vs h ───────────────────────────────────────────────────
plt.figure(figsize=(col_width, 2.5))
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

    plt.plot(H_VALUES, gamma_vals,
             marker=markers[0],
             markerfacecolor='none',
             linestyle=lines[0],
             linewidth=lw,
             color=cmap_sys(i),
             label=f'${N}$')
    plt.plot(H_VALUES, lambda_vals,
             marker=markers[1],
             markerfacecolor='none',
             linestyle=lines[3],
             linewidth=lw,
             color=cmap_sys(i))

plt.xlabel(r'$h$')
plt.ylabel(r'$\Gamma,\;\Lambda$')
leg1 = plt.legend(
    loc='best',
    ncol=1,
    handlelength=1.,    # length of the lines in legend
    columnspacing=0.4,    # horizontal space between columns
    labelspacing=0.1,
    title=f'$N$',
    title_fontsize=14
)
leg1.set_frame_on(True)
leg1.get_frame().set_edgecolor('black')
leg1.get_frame().set_linewidth(0.8)
for legline1 in leg1.legend_handles:
    legline1.set_marker('None')

plt.grid(True, which='both', linestyle=':')
plt.tight_layout(pad=0.1)

# Save figure with tight white boundary
plt.savefig("plots_APS.pdf", bbox_inches='tight', pad_inches=0.1)
plt.show()


# ─── 2) Information per scale for N=6 ─────────────────────────────────────────
plt.figure(figsize=(col_width, 2.5))
scales = None
for j, h in enumerate(INFO_H):
    path = os.path.join(
        DATA_DIR,
        f'lowEnergyStates_N8_J{J_VALUE:.3f}_h{h:.3f}.h5'
    )
    if not os.path.exists(path):
        continue
    data, attrs = load_information_per_scale(d=0, file_path=path)
    vals = np.vstack([data[field] for field in data.dtype.names]).T
    y = vals[0, :-1]
    scales = np.arange(len(y)) if scales is None else scales

    plt.plot(scales, y,
             marker=markers[j],
             markerfacecolor='none',
             linestyle=lines[j],
             linewidth=lw,
             color=cmap_info(j),
             label=fr'${h:.2f}$')

#plt.xlabel(r'$\ell$', labelpad=-8)
#plt.ylabel(r'$I(\ell)$', labelpad=-8)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$I(\ell)$')
leg2 = plt.legend(
    loc='best',
    handlelength=1.,    # length of the lines in legend
    columnspacing=0.4,    # horizontal space between columns
    labelspacing=0.1,
    title=f'$h$',
    title_fontsize=14
)
leg2.set_frame_on(True)
leg2.get_frame().set_edgecolor('black')
leg2.get_frame().set_linewidth(0.8)
for legline2 in leg2.legend_handles:
    legline2.set_marker('None')

plt.grid(True, which='both', linestyle=':')
plt.tight_layout(pad=0.1)

# Save figure with tight white boundary
plt.savefig("plots_APS.pdf", bbox_inches='tight', pad_inches=0.1)
plt.show()
