import numpy as np
from itertools import product, combinations
from functools import reduce


from qiskit.quantum_info import DensityMatrix, partial_trace
from qiskit.quantum_info import Pauli
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec


#%% Pauli matrix tools
sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

_, eigvec_x = np.linalg.eig(sigma_x)
_, eigvec_y = np.linalg.eig(sigma_y)
_, eigvec_z = np.linalg.eig(sigma_z)

projectors = {
    'X': {'+': np.outer(eigvec_x[:, 0], eigvec_x[:, 0].conj()), '-': np.outer(eigvec_x[:, 1], eigvec_x[:, 1].conj())},
    'Y': {'+': np.outer(eigvec_y[:, 0], eigvec_y[:, 0].conj()), '-': np.outer(eigvec_y[:, 1], eigvec_y[:, 1].conj())},
    'Z': {'+': np.outer(eigvec_z[:, 0], eigvec_z[:, 0].conj()), '-': np.outer(eigvec_z[:, 1], eigvec_z[:, 1].conj())}
}

#%% Functions


# SRE1 Lattice
def Xi_pure(psi: np.ndarray, remove_I=False) -> dict:

    # Pauli strings
    L = int(np.log2(psi.shape))
    pauli_strings = []
    pauli_iter = product('IXYZ', repeat=L)
    for element in pauli_iter:
        pauli_strings.append(''.join(element))
    if remove_I:
        Id = 'I' * L
        pauli_strings.remove(Id)
        norm = (2 ** L) - 1
    else:
        norm = 2 ** L

    # Probability distribution
    prob_dist = {}
    rho = DensityMatrix(psi).data
    for i, pauli_string in enumerate(pauli_strings):
        P = Pauli(pauli_string).to_matrix()
        prob_dist[pauli_string] = (np.abs(np.trace(rho @ P)) ** 2) / norm

    return prob_dist

def calc_Xi_subsystem(Xi_pure, n, l, pauli_strings) -> np.ndarray:

    # Probability distribution for the subsystem
    prob_dist_nl = np.zeros((len(pauli_strings), ))
    for i, P in enumerate(pauli_strings):
        for key in Xi_pure.keys():
            if P == key[n: n + l]:
                prob_dist_nl[i] += Xi_pure[key]

    return prob_dist_nl

def calc_Xi_all_subsystems(psi: np.ndarray) -> dict:

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

def calc_SRE1_all_subsystems(psi: np.ndarray) -> dict[np.ndarray]:

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

def calc_SRE1_lattice(psi: np.ndarray) -> dict[np.ndarray]:

    # Definition and preallocation
    assert len(psi.shape) == 1
    L = int(np.log2(psi.size))
    SRE1 = calc_SRE1_all_subsystems(psi)
    magic_latt = {}

    # Levels of the information (SRE1) lattice
    for l in range(1, L + 1):
        if l == 1:
            magic_latt[l] = l - SRE1[l]
        elif l == 2:
            magic_latt[l] = 2 - l - SRE1[l] + SRE1[l - 1][:-1] + SRE1[l - 1][1:]
        else:
            magic_latt[l] = -SRE1[l] + SRE1[l - 1][:-1] + SRE1[l - 1][1:] - SRE1[l - 2][1:-1]

    return magic_latt

def calc_total_info(SRE1_latt) -> float:
    total_info = 0
    for key in SRE1_latt.keys():
        total_info += np.sum(SRE1_latt[key])
    return total_info

def calc_SRE1_pure(psi: np.ndarray, remove_I=False) -> float:
    L = int(np.log2(psi.size))

    if remove_I:
        prob_dist = np.array(list(Xi_pure(psi, remove_I=True).values()))
        offset_STAB = 0
    else:
        prob_dist = np.array(list(Xi_pure(psi, remove_I=False).values()))
        offset_STAB = L * np.log2(2)
    prob_dist[prob_dist < 1e-16] = 1e-22
    shannon_entropy = - np.sum(prob_dist * np.log2(prob_dist)) - offset_STAB
    return shannon_entropy

def plot_probabilities(prob_dist, num_qubits, fig) -> None:

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

def measurement_outcome_allstab_shannon(rho):

    # Stabilizer operators
    L = int(np.log2(len(rho)))
    pauli_strings = []
    pauli_iter = product('XYZ', repeat=L)
    for element in pauli_iter:
        pauli_strings.append(''.join(element))
    stab_operators, list_strings = get_all_stab_operators(pauli_strings, L)
    shannon_entropies = np.zeros((len(stab_operators), ))

    # Shannon entropy for all stabiliser operator
    for i, operator in enumerate(stab_operators):
        list_projectors = get_projectors(operator)
        measurement_probs = np.array([np.trace(rho @ projector) for projector in list_projectors])
        measurement_probs[measurement_probs< 1e-16] = 1e-22
        shannon_entropies[i] = - measurement_probs @ np.log2(measurement_probs)

    return shannon_entropies, list_strings

def get_projectors(operator):
    _, eigenvecs = np.linalg.eig(operator)
    list_projectors = [np.outer(eigenvecs[:, i], eigenvecs[:, i].conj()) for i in range(len(eigenvecs[:, 0]))]
    return list_projectors

def get_all_stab_operators(pauli_strings, num_qubits):
    list_strings = list(combinations(pauli_strings, num_qubits)) + pauli_strings
    pauli_matrices = [Pauli(string).to_matrix() for string in pauli_strings]
    if num_qubits==1:
        stab_operators = pauli_matrices
    else:
        list_generators = list(combinations(pauli_matrices, num_qubits))
        list_stab_groups = list()
        stab_operators = [(1 / num_qubits) * sum(generators) for generators in list_generators] + pauli_matrices
    return stab_operators, list_strings

def measurement_outcome_shannon_product(rho):

    L = int(np.log2(len(rho)))
    pauli_strings = []
    pauli_iter = product('XYZ', repeat=L)
    for element in pauli_iter:
        pauli_strings.append(''.join(element))
    # Id = 'I' * L
    # pauli_strings.remove(Id)

    shannon_entropies = np.zeros((len(pauli_strings), ))
    for i, pauli in enumerate(pauli_strings):

        # Get all possible orthogonal projectors in the measurement basis of the specific pauli string
        projectors_1q, projectors_multiq = [], []
        for j in pauli:
            projectors_1q.append([projectors[j]['+'], projectors[j]['-']])
        projectors_multiq_tuple = product(*projectors_1q)
        for projector_tuple in projectors_multiq_tuple:
            projectors_multiq.append(reduce(lambda A, B: np.kron(A, B), projector_tuple))

        measurement_probs = np.array([np.trace(rho @ projector) for projector in projectors_multiq])
        measurement_probs[measurement_probs< 1e-16] = 1e-22
        shannon_entropies[i] = - measurement_probs @ np.log2(measurement_probs)

    return shannon_entropies, pauli_strings



# Minimising entanglement
def apply_v_pair(v_pair: list) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    for i in range(2):
        if v_pair[i]=='V':
            qc.h(i)
            qc.s(i)
        elif v_pair[i]=='W':
            qc.h(i)
            qc.s(i)
            qc.h(i)
            qc.s(i)
        else:
            pass
    return qc

def minimal_clifford_disentanglers(visualise=False) -> list[QuantumCircuit]:

    # Single qubit gates to sample from
    v_list = ["I", "W", "V"]
    v_pair_list = [[i, j] for i in v_list for j in v_list]

    # Class 1 - I
    all_circuits = [QuantumCircuit(2)]

    # Class 2 - CNOT
    for v_pair in v_pair_list:
        qc_c2 = apply_v_pair(v_pair)
        qc_c2.cx(0, 1)
        all_circuits.append(qc_c2) #.to_gate())

    # Class 3 - iSWAP
    for v_pair in v_pair_list:
        qc_c3 = apply_v_pair(v_pair)
        qc_c3.cx(1, 0)
        qc_c3.cx(0, 1)
        all_circuits.append(qc_c3) #.to_gate())

    # Class 4 - SWAP
    qc_c4 = QuantumCircuit(2)
    qc_c4.cx(0, 1)
    qc_c4.cx(1, 0)
    qc_c4.cx(0, 1)
    all_circuits.append(qc_c4) #.to_gate())

    # To gate
    all_gates = []
    for item in all_circuits:
        if visualise:
            item.draw(output="mpl", style="iqp")
            plt.show()
        all_gates.append(item.to_gate())

    return all_gates

def minimise_entanglement(psi, N, disentangling_gates, maxsweeps=10):

    # Definitions
    min_qc = QuantumCircuit(N)

    # Starting point of the minimisation
    tol_EE = 1e-10
    tol_convergence = 1e-5
    nsweeps, ntrials = 0, 0
    EE = calculate_EE_all_cuts(psi.data)
    print('EE: ', EE)

    # Minimisation Sweep
    rev = False
    while EE > tol_EE and nsweeps < maxsweeps:
        print('another sweep with rev', rev)
        # Sweep over all qubits
        for n in range(0, N - 1):

            # Order of the sweep
            if rev:
                q0 = (N - 1) - n
                q1 = (N - 1) - n - 1
            else:
                q0 = n
                q1 = n + 1

            # Find the best disentangling circuit
            print('q0: ', q0, 'q1: ', q1)
            EE_1qb = calculate_EE_sweep(psi, q0, rev)
            print('EE_cut: ', EE_1qb)
            if EE_1qb < tol_EE:
                continue

            print('---------------')
            print('q0: ', q0, 'q1: ', q1)
            print('EE_cut: ', EE_1qb)

            minimizing_gate = QuantumCircuit(N)
            for gate_idx, gate in enumerate(disentangling_gates):

                # Trial circuit
                circuit = QuantumCircuit(N)
                circuit.append(gate, [q0, q1])

                # New entanglement entropy of the cut
                psi_trial = psi.evolve(circuit)
                EE_trial = calculate_EE_sweep(psi_trial, q0, rev)
                # print('EE_trial:', EE_trial, 'E_1qb:', EE_1qb)
                if EE_trial < EE_1qb:
                    # print('Changed minimal value of EE')
                    EE_1qb = EE_trial
                    minimizing_gate = circuit
                    if EE_1qb < tol_EE:
                        break

                    # has_changed = True

            # Evolve with the minimising circuit
            min_qc.compose(minimizing_gate, inplace=True)
            print('Minimising circuit up to :', q0, q1, rev)
            print(min_qc.decompose())
            psi = psi.evolve(minimizing_gate)

        EE_new = calculate_EE_all_cuts(psi.data)
        print('Total EE: ', EE)
        if np.abs(EE - EE_new) < tol_convergence:
            break
        else:
            EE = EE_new
        print('Total EE_new: ', EE)
        rev = not rev
        nsweeps += 1
        print(nsweeps, rev)
    return psi, min_qc

def calculate_EE_sweep(psi: Statevector, q0, rev=False) -> float:

    # System size and qubits
    N = int(np.log2(len(psi.data)))  # no. qubits
    tot_qubits = [i for i in range(0, N)]

    # Tracing out the qubits to the left/right of the cut
    if q0 == N - 1 or rev:
        traced_out_qubits = tot_qubits[:q0]
    else:
        traced_out_qubits = tot_qubits[q0 + 1:]

    # Reduced density matrix of the cut
    rho = partial_trace(psi, traced_out_qubits)
    lambdas = np.linalg.eigvalsh(rho)
    lambdas[(-1e-13 < lambdas) & (lambdas < 1e-13)] = 1e-13
    S = - np.sum(lambdas * np.log2(lambdas))

    return S

def calculate_EE_all_cuts(psi: np.ndarray) -> float:

    # Definitions
    N = int(np.log2(len(psi)))
    psi_EE = psi.reshape(2 ** N, 1)
    S_cuts = []

    # Density matrix
    rho = np.outer(psi_EE, psi_EE.conj())
    rho = rho.reshape([2] * 2 * N)

    # Entropy for the cut
    for n in range(N, 1, -1):
        l = rho.ndim
        rho_partial = np.trace(rho, axis1=n - 1, axis2=l - 1)
        lambdas = np.linalg.eigvalsh(rho_partial.reshape([2 ** (n - 1), 2 ** (n - 1)]))
        lambdas[(-1e-13 < lambdas) & (lambdas<1e-13)] = 1e-13
        S = - np.sum(lambdas * np.log2(lambdas))
        S_cuts.append(S)
        rho = rho_partial

    return np.sum(S_cuts)


# Random (probably outdated)
def partial_trace_lattice(psi_full, n, l):

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
            rho_subsytem = partial_trace_lattice(psi, n, l)
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

def non_integer_magic(info_lattice):

    magic = 0
    for scale in info_lattice.keys():
        for info in info_lattice[scale]:
            magic += np.abs(info - round(info))
    return magic

