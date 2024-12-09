import numpy as np
from qiskit.quantum_info import DensityMatrix, partial_trace
from itertools import product
from qiskit.quantum_info import Pauli


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import colorbar
from matplotlib.gridspec import GridSpec


# Quantum magic lattice
def partial_trace(psi_full, n, l):

    # Vectorised density matrix
    rho_vec = np.kron(psi_full, np.conj(psi_full))
    L = int(np.log2(np.size(rho_vec))) // 2
    rho = np.reshape(rho_vec, (2 * L) * [2])

    # Subsystem density matrix
    for w in range(L - l):
        if w < n:
            rho = np.trace(rho, axis1=0, axis2=L - w)
        elif w == n or w > n:
            rho = np.trace(rho, axis1=l, axis2=l + L - w)
    dim = 2 ** (l)

    return np.reshape(rho, (dim, dim))

def Xi_mixed(rho_subsystem, pauli_strings):

    # Probability distribution
    Tr_rhoP = np.zeros((len(pauli_strings),))
    for i, pauli_string in enumerate(pauli_strings):
        P = Pauli(pauli_string).to_matrix()
        Tr_rhoP[i] = np.abs(np.trace(rho_subsystem @ P))
    Xi_mixed = (Tr_rhoP ** 2) / np.sum(Tr_rhoP ** 2)

    return Xi_mixed

def calc_SRE1(psi):

    L = int(np.log2(psi.size))
    SRE1 = {}

    for l in range(1, L + 1):

        # Pauli strings for the subsystem
        pauli_strings = []
        pauli_iter = product('IXYZ', repeat=l)
        for element in pauli_iter:
            pauli_strings.append(''.join(element))
        SRE1[l] = []

        for n in range(L - l + 1):
            # Subsystem density matrix and probability
            rho_subsytem = partial_trace(psi, n, l)
            prob_dist = Xi_mixed(rho_subsytem, pauli_strings)

            # Shannon entropy
            prob_dist[prob_dist < 1e-16] = 1e-22
            shannon_entropy = - np.sum(prob_dist * np.log(prob_dist)) - l * np.log(2)
            SRE1[l].append(shannon_entropy)

        SRE1[l] = np.array(SRE1[l])
    return SRE1

def calc_magic(psi):

    assert len(psi.shape) == 1
    L = int(np.log2(psi.size))
    SRE1 = calc_SRE1(psi)
    magic_latt = {}
    for l in range(1, L + 1):
        if l == 1:
            magic_latt[l] = l - SRE1[l]
        elif l == 2:
            magic_latt[l] = 2 - l - SRE1[l] + SRE1[l - 1][:-1] + SRE1[l - 1][1:]
        else:
            magic_latt[l] = -SRE1[l] + SRE1[l - 1][:-1] + SRE1[l - 1][1:] - SRE1[l - 2][1:-1]

    return magic_latt

def plot_magic_latt(magic_latt, ax, color_map, min_value=0., max_value=2., indicate_ints=False, tol_ints=1e-14):

    # Color normalisation
    norm = Normalize(vmin=min_value, vmax=max_value)

    # Plot
    L = max(magic_latt.keys())
    r = 1/(4*L)
    for l in magic_latt:
        for x in range(len(magic_latt[l])):
            value = magic_latt[l][x]
            if indicate_ints and np.allclose(value, round(value), atol=tol_ints) and round(value) != 0:
                linewidth = 3
                edgecolor='black'
            else:
                linewidth = 0.2
                edgecolor = 'black'
            ax.add_artist(plt.Circle((x/L+l/(2*L), (l-0.5)/L), r, facecolor=color_map(norm(value)), edgecolor=edgecolor, linewidth=linewidth))
    ax.set_xlim([-2*r,1])
    ax.set_ylim([-2*r,1+2*r])
    ax.set_aspect('equal')
    ax.axis('off')


# Classical magic lattice
def Xi_pure(psi):

    # Pauli strings
    L = int(np.log2(psi.shape))
    pauli_strings = []
    pauli_iter = product('IXYZ', repeat=L)
    for element in pauli_iter:
        pauli_strings.append(''.join(element))

    # Probability distribution
    prob_dist = {}
    rho = DensityMatrix(psi).data
    for i, pauli_string in enumerate(pauli_strings):
        P = Pauli(pauli_string).to_matrix()
        prob_dist[pauli_string] = (np.abs(np.trace(rho @ P)) ** 2) / (2 ** L)

    return prob_dist

def calc_Xi_subsystem(Xi_pure, n, l, pauli_strings):

    # Probability distribution for the subsystem
    prob_dist_nl = np.zeros((len(pauli_strings), ))
    for i, P in enumerate(pauli_strings):
        for key in Xi_pure.keys():
            if P == key[n: n + l]:
                prob_dist_nl[i] += Xi_pure[key]

    return prob_dist_nl

def calc_all_Xi(psi):

    # Preallocation
    full_prob = Xi_pure(psi)
    L = int(np.log2(psi.size))
    prob_dist = {}

    for l in range(1, L + 1):

        # Pauli strings for the subsystem
        pauli_strings = []
        pauli_iter = product('IXYZ', repeat=l)
        for element in pauli_iter:
            pauli_strings.append(''.join(element))

        # Probability distribution for each subsystem
        prob_dist[l] = {}
        for n in range(L - l + 1):
            prob_nl = calc_Xi_subsystem(full_prob, n, l, pauli_strings)
            prob_dist[l][n] = prob_nl

    return prob_dist

def calc_classical_SRE1(psi):

    # Preallocation
    full_prob = Xi_pure(psi)
    L = int(np.log2(psi.size))
    SRE1 = {}

    for l in range(1, L + 1):

        # Pauli strings for the subsystem
        pauli_strings = []
        pauli_iter = product('IXYZ', repeat=l)
        for element in pauli_iter:
            pauli_strings.append(''.join(element))
        SRE1[l] = []

        for n in range(L - l + 1):
            # Subsystem density matrix and probability
            prob_dist = calc_Xi_subsystem(full_prob, n, l, pauli_strings)

            # Shannon entropy
            prob_dist[prob_dist < 1e-16] = 1e-22
            shannon_entropy = - np.sum(prob_dist * np.log2(prob_dist)) - l * np.log2(2)
            SRE1[l].append(shannon_entropy)

        SRE1[l] = np.array(SRE1[l])
    return SRE1

def calc_classical_magic(psi):

    # Definition and preallocation
    assert len(psi.shape) == 1
    L = int(np.log2(psi.size))
    SRE1 = calc_classical_SRE1(psi)
    magic_latt = {}

    # Levels of the information (magic) lattice
    for l in range(1, L + 1):
        if l == 1:
            magic_latt[l] = l - SRE1[l]
        elif l == 2:
            magic_latt[l] = 2 - l - SRE1[l] + SRE1[l - 1][:-1] + SRE1[l - 1][1:]
        else:
            magic_latt[l] = -SRE1[l] + SRE1[l - 1][:-1] + SRE1[l - 1][1:] - SRE1[l - 2][1:-1]

    return magic_latt

def calc_total_info(magic_latt):

    total_info = 0
    for key in magic_latt.keys():
        total_info += np.sum(magic_latt[key])
    return total_info

def shannon(psi):
    L = int(np.log2(psi.size))
    prob_dist = np.array(list(Xi_pure(psi).values()))
    prob_dist[prob_dist < 1e-16] = 1e-22
    shannon_entropy = - np.sum(prob_dist * np.log2(prob_dist)) - L * np.log2(2)
    return shannon_entropy

def plot_probabilities(prob_dist, num_qubits, fig):

    gs = GridSpec(num_qubits, num_qubits, figure=fig, hspace=0.25, wspace=0.25)
    for l in prob_dist.keys():

        # Pauli strings for the subsystem
        pauli_strings = []
        pauli_iter = product('IXYZ', repeat=l)
        for element in pauli_iter:
            pauli_strings.append(''.join(element))


        for n in prob_dist[l].keys():
            ax = fig.add_subplot(gs[num_qubits - l, n])
            n_paulis = len(prob_dist[l][n])
            ax.plot(np.arange(n_paulis), prob_dist[l][n], 'o', color='Blue')

            for i, string in enumerate(pauli_strings):
                if prob_dist[l][n][i] > 1e-10:
                    ax.text(i + len(pauli_strings) / 50, prob_dist[l][n][i], f'{string}')

            ax.set_ylim(bottom=0)
            ax.set_ylabel('$\Xi$', fontsize='20')
            ax.set_xlabel('$\sigma$', fontsize='20')





