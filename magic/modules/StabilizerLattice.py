import numpy as np
from itertools import product, combinations
from functools import reduce


from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Pauli, PauliList
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

#%% Logging setup
logger_stab = logging.getLogger('stab')
logger_stab.setLevel(logging.INFO)

stream_handler = colorlog.StreamHandler()
formatter = ColoredFormatter(
    '%(black)s%(asctime) -5s| %(blue)s%(name) -10s %(black)s| %(cyan)s %(funcName) '
    '-40s %(black)s|''%(log_color)s%(levelname) -10s | %(message)s',
    datefmt=None,
    reset=True,
    log_colors={
        'TRACE': 'black',
        'DEBUG': 'purple',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

stream_handler.setFormatter(formatter)
logger_stab.addHandler(stream_handler)

#%% Pauli matrices
sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


#%% Functions

# Calculating the stabilizer von Neumann entropy
def stabilizer_vN_entropy(rho, return_entropy_list=False):

    # Pauli strings
    L = int(np.log2(len(rho)))
    pauli_strings = []
    pauli_iter = product('IXYZ', repeat=L)
    for element in pauli_iter:
        pauli_strings.append(''.join(element))
    pauli_strings.remove('I' * L)

    # All possible stabilizer group generators
    logger_stab.info('Calculating all possible stabilizer group generators...')
    list_groups, list_strings = get_all_stab_generators(pauli_strings, L)    # [[g1, g2, ..., gn], [g1', g2', ..., gn'], ...]
    shannon_entropies = np.zeros((len(list_groups), ))

    # Shannon entropy for all stabiliser operator
    logger_stab.info('Calculating shannon entropies...')
    for i, group in enumerate(list_groups):

        # List all projectors in the group
        list_all_projectors = []  # [[P1p, P1m], [P2p, P2m], ....]
        for generator in group:
            list_all_projectors.append(get_projectors(generator))

        # Calculate probabilities of the different possible simultaneous projective measurements of the stabilizer generators
        if L==1:
            measurement_probs = np.array([np.trace(rho @ projector) for projector in list_all_projectors[0]])
        else:
            list_combined_projectors = list(product(*list_all_projectors))   # [[P1p, P2p, ...], [P1p, P2m, ], ...]
            projective_measurements = [reduce(lambda A, B: A @ B, projector_tuple) for projector_tuple in list_combined_projectors]   # [P1p@P2p@..., P1p@P2m..., ...]

            projectors_square = [P @ P - P for P in projective_measurements]
            if not np.allclose(projectors_square, 0., 1e-10):
                raise ValueError('Projectors are not really projectors')

            measurement_probs = np.array([np.trace(rho @ projector) for projector in projective_measurements])

            if np.allclose(np.imag(measurement_probs), 0., 1e-10):
                measurement_probs = np.real(measurement_probs)
            else:
                raise ValueError('Probabilities are complex')

            if any(x > 1.00000001 for x in measurement_probs):
                raise ValueError('Probability is greater than 1')

            if any(x < -1e-10 for x in measurement_probs):
                raise ValueError('Probability is negative')

            measurement_probs /= np.sum(measurement_probs)
            if not np.allclose(np.sum(measurement_probs), 1., 1e-10):
                raise ValueError('Probabilities not adding up to 1')

        # Calculate shannon entropy
        measurement_probs[measurement_probs< 1e-16] = 1e-22
        shannon_entropies[i] = - measurement_probs @ np.log2(measurement_probs)

    if return_entropy_list:
        return np.min(shannon_entropies), shannon_entropies, list_strings
    else:
        return np.min(shannon_entropies)

def get_projectors(operator):

    # Eigenspaces of the Pauli string
    eigvals, eigvecs = np.linalg.eig(operator)
    eigspace_p = np.where(eigvals > 0)[0]
    eigspace_m = np.where(eigvals < 0)[0]

    # Hard check, to be improved in the future
    for (i, j) in combinations(eigspace_p, 2):
        if not np.allclose(eigvecs[:, i] @ eigvecs[:, j].conj(), 0., 1e-10):
            print(eigvecs[:, i] @ eigvecs[:, j].conj())
            raise ValueError('The eigenvectors of the Pauli strings are not linearly independent!')

    # Full projectors on each eigenspace
    projectors_p = [np.outer(eigvecs[:, i], eigvecs[:, i].conj()) for i in eigspace_p]
    projectors_m = [np.outer(eigvecs[:, i], eigvecs[:, i].conj()) for i in eigspace_m]
    P_p = reduce(lambda A, B: A + B, projectors_p)
    P_m = reduce(lambda A, B: A + B, projectors_m)

    return [P_p, P_m]

def get_all_stab_generators(pauli_strings, num_qubits):

    # All subsets of n Pauli strings
    list_strings = list(combinations(pauli_strings, num_qubits))
    list_groups, group_strings = [], []

    # Check which subsets are stabilizer groups
    logger_stab.info('Checking commutation of possible group elements...')
    for group in list_strings:
        flag = True
        group_str = [''] * num_qubits

        for element in group:
            group_str = [x + y for x, y in zip(element, group_str)]
            if np.prod(Pauli(element).commutes(PauliList(group)))==0:
                flag = False
                break
            else:
                pass
        if flag:
            for pauli_1q in group_str:
                pauli_lst = PauliList(list(pauli_1q)).to_matrix()
                pauli_matrix = reduce(lambda x, y: x @ y, pauli_lst)
                if np.allclose(pauli_matrix, np.eye(2)):
                    flag = False
        if flag:
            list_groups.append([Pauli(string).to_matrix() for string in group])
            group_strings.append(group)


    return list_groups, group_strings

def partial_trace(rho, n, l):
    rho_vec = np.array(rho.flatten())
    L = int(np.log2(np.size(rho_vec))) // 2
    rho = np.reshape(rho_vec, (2 * L) * [2])
    for w in range(L - l):
        if w < n:
            rho = np.trace(rho, axis1=0, axis2=L - w)
        elif w == n or w > n:
            rho = np.trace(rho, axis1=l, axis2=l + L - w)
    dim = 2 ** l
    return np.reshape(rho, (dim, dim))

# Calculating the stabilizer von Neumann entropy lattice
def calc_stab_vN_entropies(psi):
    rho = np.outer(psi, psi.conj())
    L = int(np.log2(psi.size))
    S = {}

    for l in range(1, L + 1):
        S[l] = []
        for n in range(L - l + 1):
            logger_stab.info(f'Calculating subsystem l: {l}, n: {n}')
            reduced_rho = partial_trace(rho, n, l)
            subsystem_S = stabilizer_vN_entropy(reduced_rho)
            S[l].append(subsystem_S)
        S[l] = np.array(S[l])
    return S

def calc_stabilizer_vN_info(psi):

    assert len(psi.shape) == 1
    L = int(np.log2(psi.size))
    S = calc_stab_vN_entropies(psi)
    info_latt = {}
    for l in range(1, L + 1):
        if l == 1:
            info_latt[l] = l - S[l]
        elif l == 2:
            info_latt[l] = 2 - l - S[l] + S[l - 1][:-1] + S[l - 1][1:]
        else:
            info_latt[l] = -S[l] + S[l - 1][:-1] + S[l - 1][1:] - S[l - 2][1:-1]

    return info_latt

def plot_stabilizer_vN_latt(info_latt, ax, color_map, min_value=0, max_value=2., indicate_ints=False, tol_ints=1e-14):

    # Color normalisation
    norm = Normalize(vmin=min_value, vmax=max_value)

    # Plot
    L = max(info_latt.keys())
    r = 1/(4*L)
    for l in info_latt:
        for x in range(len(info_latt[l])):
            value = info_latt[l][x]
            if indicate_ints and np.allclose(value, round(value), atol=tol_ints) and round(value) != 0:
                linewidth = 3
                edgecolor= 'black'
            elif value < -1e-10:
                linewidth = 3
                edgecolor = 'blue'
            else:
                linewidth = 0.2
                edgecolor = 'black'
            ax.add_artist(plt.Circle((x/L+l/(2*L), (l-0.5)/L), r, facecolor=color_map(norm(value)), edgecolor=edgecolor, linewidth=linewidth))
    ax.set_xlim([-2*r,1])
    ax.set_ylim([-2*r,1+2*r])
    ax.set_aspect('equal')
    ax.axis('off')

