import numpy as np
import random as rnd
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from itertools import product
import qiskit  # version 1.2.4
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library.standard_gates import HGate, SGate, TGate, CXGate
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info import Statevector,  SparsePauliOp, Pauli, StabilizerState, Operator, random_clifford, DensityMatrix, partial_trace
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.primitives import StatevectorEstimator

params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "axes.facecolor" : "white",
          "xtick.direction" : "in",
          "ytick.direction" : "in",
          "text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern Serif"],
          "figure.facecolor":  (1.0, 0.0, 0.0, 0.0),
          "savefig.facecolor": (0.0, 0.0, 1.0, 0.0)}
plt.rcParams.update(params)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}'
                              r'\usepackage{amssymb}'
                              r' \usepackage[version=4]{mhchem}'
                              r'\usepackage{gensymb}'
                             r'\usepackage[braket]{qcircuit}')

def disentangle_greedy_search():
    fig = plt.figure(figsize=(6.4*2, 4.8))
    gs = gridspec.GridSpec(nrows=2, ncols=2)
    ax0 = fig.add_subplot(gs[0:2, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    Ns = [2, 3, 4, 5, 6]  # fine up to 7 qubits, too slow otherwise
    viridis = plt.cm.viridis
    colors = viridis(np.linspace(0.1, 0.9, len(Ns)))
    discrete_cmap = mcolors.ListedColormap(colors, name='viridis_discrete')
    colors = list(discrete_cmap.colors)
    tmax = 10  # doesn't really matter
    runs = 1

    for i, N in enumerate(Ns):
        for j in range(runs):
            EE, t, magic, ee_ent = simulation_random_greedy_v2(N, tmax)
            steps_magic = [i for i in range(len(list(magic)))]
            if j == 0:
                ax0.plot(t / N, EE, marker=".", label=f"$N=$ {{{N}}}", color=colors[i], linewidth=0.75, alpha=0.75)
                ax1.plot(steps_magic, magic, marker="*", label=f"$N=$ {{{N}}}", color=colors[i], linewidth=0.75, alpha=0.75)
                ax2.plot(steps_magic, ee_ent, marker=".", label=f"$N=$ {{{N}}}", color=colors[i], linewidth=0.75, alpha=0.75)
            else:
                ax0.plot(t / N, EE, marker=".", color=colors[i], linewidth=0.75, alpha=0.75)
                ax1.plot(steps_magic, magic, marker="*", color=colors[i], linewidth=0.75, alpha=0.75)
                ax2.plot(steps_magic, ee_ent, marker=".", color=colors[i], linewidth=0.75, alpha=0.75)

    ax0.set_xlabel(r"$t/N$"), ax0.set_ylabel(r"$\mathcal{E}(\ket{\psi})$"), ax0.legend()
    ax0.set_title(r"Clifford + T, sum($\{S \}_{\text{cuts}}$) check")
    ax1.set_xlabel(r"''$t$''"), ax1.set_ylabel(r"$\mathcal{M}(\ket{\psi})$"), ax1.legend()
    ax1.set_title(r"SRE during entangling phase")
    ax2.set_xlabel(r"''$t$''"), ax2.set_ylabel(r"$\mathcal{E}(\ket{\psi})$"), ax1.legend()
    ax2.set_title(r"EE during entangling phase")

    fig.savefig("disentangle_greedy.pdf", bbox_inches="tight")

    plt.show()

def disentangle_bell():  # bell and GHZ states for testing
    fig = plt.figure(figsize=(6.4, 4.8))
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0])

    viridis = plt.cm.viridis
    colors = viridis(np.linspace(0.1, 0.9, 10))
    discrete_cmap = mcolors.ListedColormap(colors, name='viridis_discrete')
    colors = list(discrete_cmap.colors)
    alt_colors = plt.cm.plasma(np.linspace(0.1, 0.9, 10))
    alt_discrete_cmap = mcolors.ListedColormap(alt_colors, name='plasma_discrete')
    alt_colors = list(alt_discrete_cmap.colors)
    ls = ["--", "-", "dotted"]

    N_bell = 2
    psi0_bell = Statevector.from_label("0" * N_bell)
    qc_bell = QuantumCircuit(N_bell)
    qc_bell.h(0)
    qc_bell.cx(0, 1)
    psi_bell = psi0_bell.evolve(qc_bell)
    EE0_bell, sre = calculate_EE_gen(psi_bell.data)
    EE0_bell_0 = calculate_EE_sweep(psi_bell, 0)
    EE0_bell_1 = calculate_EE_sweep(psi_bell, 1)
    print(f"EE_bell_0: {EE0_bell_0}, EE_bell_1: {EE0_bell_1}")
    print(f"EE_0_bell = {EE0_bell}")

    N_ghz = 3
    psi0_ghz = Statevector.from_label("0" * N_ghz)
    qc_ghz = QuantumCircuit(N_ghz)
    qc_ghz.h(0)
    qc_ghz.cx(0, 1)
    qc_ghz.cx(1, 2)
    psi_ghz = psi0_ghz.evolve(qc_ghz)
    EE0_ghz, sre = calculate_EE_gen(psi_ghz.data)
    EE0_ghz_0 = calculate_EE_sweep(psi_ghz, 0)
    EE0_ghz_1 = calculate_EE_sweep(psi_ghz, 1)
    EE0_ghz_2 = calculate_EE_sweep(psi_ghz, 2)
    print(f"EE_ghz_0: {EE0_ghz_0}, EE_ghz_1: {EE0_ghz_1}, EE_ghz_2: {EE0_ghz_2}")
    #qc_ghz_rev = qc_ghz.inverse()
    qc_ghz_rev = QuantumCircuit(N_ghz)
    qc_ghz_rev.cx(1, 2)
    psi_ghz_rev_1 = psi_ghz.evolve(qc_ghz_rev)
    qc_ghz_rev.cx(0, 1)
    psi_ghz_rev_2 = psi_ghz.evolve(qc_ghz_rev)
    qc_ghz_rev.h(0)
    psi_ghz_rev = psi_ghz.evolve(qc_ghz_rev)
    EE0_ghz_rev_1 = calculate_EE_sweep(psi_ghz_rev_1, 1)
    EE0_ghz_rev_2 = calculate_EE_sweep(psi_ghz_rev_2, 0)
    EE0_ghz_rev, sre = calculate_EE_gen(psi_ghz_rev.data)
    print(f"EE_0_ghz = {EE0_ghz}, cx1 = {EE0_ghz_rev_1}, cx2 = {EE0_ghz_rev_2}, reverse = {EE0_ghz_rev}")
    #qc_ghz.draw(output="mpl", style="iqp")
    #qc_ghz_rev.draw(output="mpl", style="iqp")

    runs = 1
    for j in range(runs):
        psis_bell, best_circuit_bell = find_best_circuit(psi_bell, N_bell)
        psis_ghz, best_circuit_ghz = find_best_circuit(psi_ghz, N_ghz)
        best_circuit_bell.draw(output="mpl", style="iqp")
        best_circuit_ghz.draw(output="mpl", style="iqp")
        t_bell = []
        EE_lst_bell = []
        t_ghz = []
        EE_lst_ghz = []
        for idx, _psi in enumerate(psis_bell):
            EE, sre = calculate_EE_gen(_psi.data)
            EE_lst_bell.append(EE)
            t_bell.append(idx)
        for idx, _psi in enumerate(psis_ghz):
            EE, sre = calculate_EE_gen(_psi.data)
            EE_lst_ghz.append(EE)
            t_ghz.append(idx)

        t_bell = np.array(t_bell)
        EE_lst_bell = np.array(EE_lst_bell)
        t_ghz = np.array(t_ghz)
        EE_lst_ghz = np.array(EE_lst_ghz)
        lstyle = rnd.choice(ls)
        if j == 0:
            ax0.plot(t_bell / N_bell, EE_lst_bell, marker=".", label=f"$N=$ {{{N_bell}}} (Bell)", color=colors[j], linewidth=0.75, alpha=0.75, linestyle=lstyle)
            ax0.plot(t_ghz / N_ghz, EE_lst_ghz, marker="*", label=f"$N=$ {{{N_ghz}}} (GHZ)", color=alt_colors[j],
                     linewidth=0.75, alpha=0.75, linestyle=lstyle)
            #ax1.plot(t, SRE, marker=".", label=f"$N=$ {{{N}}}", color=colors[i], linewidth=0.75)
        else:
            ax0.plot(t_bell / N_bell, EE_lst_bell, marker=".", color=colors[j], linewidth=0.75, alpha=0.75, linestyle=lstyle)
            ax0.plot(t_ghz / N_ghz, EE_lst_ghz, marker="*", color=alt_colors[j],
                     linewidth=0.75, alpha=0.75, linestyle=lstyle)
            #ax1.plot(t, SRE, marker=".", color=colors[i], linewidth=0.75)

    ax0.set_xlabel(r"$t/N$"), ax0.set_ylabel(r"$\mathcal{E}(\ket{\psi})$"), ax0.legend()
    #ax1.set_xlabel(r"$t$"), ax1.set_ylabel(r"$\mathcal{M}(\ket{\psi})$"), ax1.legend()
    fig.savefig("disentangle_bell_cuts_v2.pdf", bbox_inches="tight")

    plt.show()


def simulation_random_greedy_v2(N, tmax):  # tmax no. "steps" of Clifford ops in disentangling circuit
    psi0 = Statevector.from_label("0" * N)  # is a stabilizer state
    entangled = False
    while not entangled:
        magic = [0]
        ee_entangling = [0]
        qc = qiskit.QuantumCircuit(N)
        rc_copy = qiskit.QuantumCircuit(N)
        psi0_test = Statevector.from_label("0" * N)
        for i in range(N-1):  # settings for number of t-gates and length of random clifford circuit
            random_circuit = random_clifford_circuit(N, N+1, max_operands=2)
            qc.compose(random_circuit, inplace=True)
            rc_copy.compose(random_circuit, inplace=True)
            which_t = rnd.randint(0, N - 1)
            qc.barrier(), qc.t(which_t), qc.barrier()

            # just for checking the evolution during entanglemment phase
            t_circuit = qiskit.QuantumCircuit(N)
            t_circuit.compose(random_circuit, inplace=True)
            t_circuit.t(which_t)
            psi0_test = psi0_test.evolve(t_circuit)
            ee, sre = calculate_EE_gen(psi0_test.data)
            magic.append(sre)
            ee_entangling.append(ee)
        psi = psi0.evolve(qc)
        EE_init, SRE_init = calculate_EE_gen(psi.data)
        #SRE_init = stabiliser_Renyi_entropy_pure(psi, 2, N)
        if EE_init > 0.5:  # arbitrary value
            entangled = True
            print(f"EE_init = {EE_init}")

    #qc.draw(output="mpl", style="iqp")
    qc_rev = rc_copy.inverse()
    psi_rev = psi.evolve(qc_rev)
    EE_rev, SRE_rev = calculate_EE_gen(psi_rev.data)
    print("nqubits: ", N, ", Entanglement entropy after reversal: ", EE_rev, ", SRE: ", SRE_rev)

    psis, min_circuit = find_best_circuit_old(psi, N, tmax)
    t_range = []
    t = 0
    EE = []
    for _psi in psis:
        _EE, SRE = calculate_EE_gen(_psi.data)
        t_range.append(t), EE.append(_EE)
        t += 1

    return np.array(EE), np.array(t_range), np.array(magic), np.array(ee_entangling)


def find_best_circuit(psi, N, tmax=5):
    h = ["I", "H"]
    v = ["I", "HS", "V"]  # also called V and W, axis swap group; x-y-z -> y-z-x -> z-x-y
    p = ["I", "X", "Y", "Z"]


    psis = [psi]

    ## unnecessary step -- we don't use them
    p_gates = []
    h_gates = []
    v_gates = []
    for i in range(0, 2):
        #p_gates.append(rnd.choice(p))
        #h_gates.append(rnd.choice(h))
        #v_gates.append(rnd.choice(v))
        p_gates.append("I")
        h_gates.append("I")
        v_gates.append("I")

    min_qc = QuantumCircuit(N)

    v_prime_list = [[i, j] for i in v for j in v]  # all possible combos of v'_0 (x) v'_1

    maxsweeps = tmax  # one sweep = one check (one way) for entire circuit
    nsweeps = 0
    ntrials = 0
    EE, sre = calculate_EE_gen(psi.data)  # initial entanglement entropy
    print(f"EE_gen = {EE}")
    no_change_count = 0
    no_change = False
    while EE > 1e-1 and nsweeps < maxsweeps and not no_change:
        rev = bool(nsweeps % 2)  # false (normal) for odd sweep (first, third, etc)
        has_changed = False
        for n in range(0, N - 1):
            print(f"trial: {ntrials}")

            all_circuits = []  # should be 20

            if not rev:  # not reversed; normal order
                q0 = n
                q1 = n + 1
            else:
                q0 = (N - 1) - n
                q1 = (N - 1) - n - 1

            ## Class 1 ## (actually identity here)
            qc_c1 = QuantumCircuit(N)
            all_circuits.append(qc_c1)

            ## Class 2 - CNOT ##
            ## CNOT on the two random qubits
            # same gates as class 1 + all combos of v_1', v_2'
            c2_circuits = []
            for v_prime in v_prime_list:
                qc_c2_v = QuantumCircuit(N)
                qc_c2_v.cx(q0, q1)
                v_operations = gate_seq_from_row(v_prime, q0, q1, N)
                qc_c2_v.compose(v_operations, inplace=True)
                c2_circuits.append(qc_c2_v)
                all_circuits.append(qc_c2_v)

            ## Class 3 - iSWAP ##
            ## CNOT_(01)CNOT_(10) on the two random qubits
            # same gates as class 2
            c3_circuits = []
            for v_prime in v_prime_list:
                qc_c3_v = QuantumCircuit(N)
                qc_c3_v.cx(q1, q0)
                qc_c3_v.cx(q0, q1)
                v_operations = gate_seq_from_row(v_prime, q0, q1, N)
                qc_c3_v.compose(v_operations, inplace=True)
                c3_circuits.append(qc_c3_v)
                all_circuits.append(qc_c3_v)

            ## Class 4 - SWAP ##
            # CNOT_(01)CNOT_(10)CNOT_(01) on the two random qubits
            # same gates as class 1
            qc_c4 = QuantumCircuit(N)
            qc_c4.cx(q0, q1)
            qc_c4.cx(q1, q0)
            qc_c4.cx(q0, q1)
            all_circuits.append(qc_c4)

            minimizing_circuit = gate_seq_from_row(["I", "I"], q0, q1, N)
            EE_1qb = calculate_EE_sweep(psi, q0, rev)
            for c_idx, circuit in enumerate(all_circuits):
                #print(circuit)
                #circuit.inverse()
                psi_trial = psi.evolve(circuit)
                EE_trial = calculate_EE_sweep(psi_trial, q0, rev)
                if EE_trial < EE_1qb:  # NOTE: sometimes they differ by very little (likely only numerical artefact) -- how should one handle this?
                    print(f"N: {N}, EE_trial = {EE_trial}, sweep no. {nsweeps}, circuit no. {c_idx}")
                    EE_1qb = EE_trial
                    minimizing_circuit = circuit
                    has_changed = True
            min_qc.compose(minimizing_circuit, inplace=True)
            psi = psi.evolve(minimizing_circuit)
            EE, sre = calculate_EE_gen(psi.data)
            psis.append(psi)
            ntrials += 1
        if not has_changed:
            print("hasn't changed")
            no_change_count += 1
            if no_change_count > 1:  # terminate if back + forth sweep haven't produced any change
                no_change = True
        nsweeps += 1
    return psis, min_qc

def find_best_circuit_old(psi, N, tmax=5):  # old version with order of operations inverted. works better at disentangling somehow but seems wrong?
    h = ["I", "H"]
    v = ["I", "HS", "V"]
    p = ["I", "X", "Y", "Z"]

    psis = [psi]

    p_gates = []
    h_gates = []
    v_gates = []
    for i in range(0, 2):
        #p_gates.append(rnd.choice(p))
        #h_gates.append(rnd.choice(h))
        #v_gates.append(rnd.choice(v))
        p_gates.append("I")
        h_gates.append("I")
        v_gates.append("I")

    min_qc = QuantumCircuit(N)

    v_prime_list = [[i, j] for i in v for j in v]  # all possible combos of v'_0 (x) v'_1

    maxsweeps = tmax  # one sweep = one check (one way) for entire circuit
    nsweeps = 0
    ntrials = 0
    EE, sre = calculate_EE_gen(psi.data)  # initial entanglement entropy
    print(f"EE_gen = {EE}")
    no_change_count = 0
    no_change = False
    while EE > 1e-1 and nsweeps < maxsweeps and not no_change:
        rev = bool(nsweeps % 2)  # false (normal) for odd sweep (first, third, etc)
        has_changed = False
        for n in range(0, N - 1):
            print(f"trial: {ntrials}")

            all_circuits = []  # should be 20

            if not rev:  # not reversed; normal order
                q0 = n
                q1 = n + 1
            else:
                q0 = (N - 1) - n
                q1 = (N - 1) - n - 1

            ## Class 1 ##
            qc_c1 = QuantumCircuit(N)
            for row in [p_gates, v_gates, h_gates]:
                operations = gate_seq_from_row(row, q0, q1, N)
                qc_c1.compose(operations, inplace=True)
            all_circuits.append(qc_c1)

            ## Class 2 - CNOT ##
            ## CNOT on the two random qubits
            # same gates as class 1 + all combos of v_1', v_2'
            qc_c2_init = QuantumCircuit(N)
            qc_c2_init.compose(gate_seq_from_row(p_gates, q0, q1, N), inplace=True)  # apply p-gates first
            v_h_gates = [v_gates, h_gates]
            c2_circuits = []
            for v_prime in v_prime_list:
                v_operations = gate_seq_from_row(v_prime, q0, q1, N)
                qc_c2_v = qc_c2_init.compose(v_operations)
                qc_c2_v.cx(q0, q1)
                for row in v_h_gates:
                    qc_c2_v.compose(gate_seq_from_row(row, q0, q1, N), inplace=True)
                c2_circuits.append(qc_c2_v)
                all_circuits.append(qc_c2_v)

            ## Class 3 - iSWAP ##
            ## CNOT_(01)CNOT_(10) on the two random qubits
            # same gates as class 2
            qc_c3_init = QuantumCircuit(N)
            qc_c3_init.compose(gate_seq_from_row(p_gates, q0, q1, N), inplace=True)
            c3_circuits = []
            for v_prime in v_prime_list:
                v_operations = gate_seq_from_row(v_prime, q0, q1, N)
                qc_c3_v = qc_c3_init.compose(v_operations)  # new circuit here!
                qc_c3_v.cx(q1, q0)
                qc_c3_v.cx(q0, q1)
                for row in v_h_gates:
                    qc_c3_v.compose(gate_seq_from_row(row, q0, q1, N), inplace=True)
                c3_circuits.append(qc_c3_v)
                all_circuits.append(qc_c3_v)

            ## Class 4 - SWAP ##
            # CNOT_(01)CNOT_(10)CNOT_(01) on the two random qubits
            # same gates as class 1
            qc_c4 = QuantumCircuit(N)
            qc_c4.compose(gate_seq_from_row(p_gates, q0, q1, N), inplace=True)
            qc_c4.cx(q0, q1)
            qc_c4.cx(q1, q0)
            qc_c4.cx(q0, q1)
            for row in v_h_gates:
                qc_c4.compose(gate_seq_from_row(row, q0, q1, N), inplace=True)
            all_circuits.append(qc_c4)

            minimizing_circuit = gate_seq_from_row(["I", "I"], q0, q1, N)
            EE_1qb = calculate_EE_sweep(psi, q0, rev)
            for c_idx, circuit in enumerate(all_circuits):
                #print(circuit)
                psi_trial = psi.evolve(circuit)
                EE_trial = calculate_EE_sweep(psi_trial, q0, rev)
                if EE_trial < EE_1qb:  # NOTE: sometimes they differ by very little (likely only numerical artefact) -- how should one handle this?
                    print(f"N: {N}, EE_trial = {EE_trial}, sweep no. {nsweeps}, circuit no. {c_idx}")
                    EE_1qb = EE_trial
                    minimizing_circuit = circuit
                    has_changed = True
            min_qc.compose(minimizing_circuit, inplace=True)
            psi = psi.evolve(minimizing_circuit)
            EE, sre = calculate_EE_gen(psi.data)
            psis.append(psi)
            ntrials += 1
        if not has_changed:
            no_change_count += 1
            if no_change_count > 1:  # terminate if back + forth sweep haven't produced any change
                no_change = True
        nsweeps += 1
    return psis, min_qc

def gate_seq_from_row(row, q0, q1, N):
    qc_0 = gate_from_str(row[0], q0, N)
    qc_1 = gate_from_str(row[1], q1, N)
    qc_0.compose(qc_1, inplace=True)
    #qc_0.draw(output="mpl", style="iqp")
    #plt.show()
    return qc_0

def gate_from_str(str, q, N):
    qc = QuantumCircuit(N)
    single_gates = list(str)
    for gate in single_gates:
        if gate == "H":
            qc.h(q)
        elif gate == "S":
            qc.s(q)
        elif gate == "X":
            qc.x(q)
        elif gate == "Y":
            qc.y(q)
        elif gate == "Z":
            qc.z(q)
        elif gate == "V":
            qc.h(q)
            qc.sdg(q)
    return qc

def calculate_EE_gen(psi):  # Note: take psi.data as input
    S_cuts = []
    N = int(np.log2(len(psi)))  # no. qubits
    psi.shape = (2 ** N, 1)  # make into column vector
    rho = np.outer(psi, psi.conj())  # full density matrix, shape 2^N x 2^N
    SRE = stabiliser_Renyi_entropy_mixed(rho, N)
    rho = rho.reshape([2] * 2 * N)
    for n in range(N, 1, -1):  ## always trace out the last index
        #contraction = generate_contraction(N_cut)
        #rho_partial = np.einsum(contraction, rho)  # trace out other subsystem
        l = rho.ndim
        rho_partial = np.trace(rho, axis1=n - 1, axis2=l - 1)
        lambdas = np.linalg.eigvalsh(rho_partial.reshape([2 ** (n - 1), 2 ** (n - 1)]))  # new matrix has shape 2 ** (n - 1) x 2 ** (n - 1)
        #print("cut: ", N - N_cut + 1, ", contraction: ", contraction, ", lambdas: ", lambdas)
        S = 0
        for lamda in lambdas[np.abs(np.array(lambdas)) > 1e-15]:  # remove very small negative values due to numerical error
            _S = lamda * np.log2(lamda)  # von Neumann entropy
            S -= _S
            if math.isnan(_S):
                print("nan lambda: ", lamda)
        S_cuts.append(S)
        rho = rho_partial
    return sum(S_cuts), SRE

def calculate_EE_sweep(psi, q0, rev=False):  # specifically for the sweep search
    N = int(np.log2(len(psi.data)))  # no. qubits
    #psi.shape = (2 ** N, 1)
    #rho = np.outer(psi, psi.conj())
    #rho = rho.reshape([2] * 2 * N)
    #for i in range(N - 1, q0, -1):
        #l = rho.ndim
        #rho = np.trace(rho, axis1=i, axis2=l-1)
    #dim_fin = rho.ndim // 2
    #rho = rho.reshape(2 ** dim_fin, 2 ** dim_fin)
    tot_qubits = [i for i in range(0, N)]
    if q0 == N - 1 or rev:  # first condition just for testing
        traced_out_qubits = tot_qubits[:q0]
    else:
        traced_out_qubits = tot_qubits[q0+1:]
    rho = partial_trace(psi, traced_out_qubits)
    S = 0
    lambdas = np.linalg.eigvalsh(rho)
    for lamda in lambdas[np.abs(np.array(lambdas)) > 1e-15]:  # remove very small negative values due to numerical artefacts
        _S = lamda * np.log2(lamda)  # von Neumann entropy
        S -= _S
        if math.isnan(_S):
            print("nan lambda: ", lamda)
    return S

## not my functions below ##

def stabiliser_Renyi_entropy_mixed(rho, num_qubits):

    # Pauli strings
    pauli_strings = []
    pauli_iter = product('IXYZ', repeat=num_qubits)
    for element in pauli_iter:
        pauli_strings.append(''.join(element))

    # Probability distribution
    Tr_rhoP = np.zeros((len(pauli_strings),))
    for i, pauli_string in enumerate(pauli_strings):
        P = Pauli(pauli_string).to_matrix()
        Tr_rhoP[i] = np.abs(np.trace(rho @ P))

    # Renyi entropy
    M2_rho = - np.log(np.sum(Tr_rhoP ** 4) / np.sum(Tr_rhoP ** 2))
    return M2_rho

def random_clifford_circuit(num_qubits, depth, max_operands=2, seed=None):
    """Generate random circuit of arbitrary size and form.
    This function will generate a random circuit by randomly selecting gates
    from the set of standard gates in :mod:`qiskit.extensions`. For example:
    .. jupyter-execute::
        from qiskit.circuit.random import random_circuit
        circ = random_circuit(2, 2, measure=True)
        circ.draw(output='mpl')
    Args:
        num_qubits (int): number of quantum wires
        depth (int): layers of operations (i.e. critical path length)
        max_operands (int): maximum operands of each gate (between 1 and 3)
        measure (bool): if True, measure all qubits at the end
        conditional (bool): if True, insert middle measurements and conditionals
        reset (bool): if True, insert middle resets
        seed (int): sets random seed (optional)
    Returns:
        QuantumCircuit: constructed circuit
    Raises:
        CircuitError: when invalid options given
    """
    if max_operands < 1 or max_operands > 3:
        raise CircuitError("max_operands must be between 1 and 3")

    # List of Clifford gates
    one_q_ops = [HGate, SGate]
    two_q_ops = [CXGate]

    # Initialise circuit and seed
    qr = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(num_qubits)
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    # Apply arbitrary random operations at every depth
    for _ in range(depth):

        # Choose either 1, 2 qubits for the operation
        remaining_qubits = list(range(num_qubits))
        while remaining_qubits:

            # Seeing if there are possible neighbour pairs remaining
            diff = 100
            allow_2qubit = False
            for q1 in remaining_qubits:
                for q2 in remaining_qubits:
                    diff = np.abs(q1 - q2)
                    if diff == 1:
                        allow_2qubit = True
                        break
                if diff == 1:
                    break

            # Selecting number of qubits to apply the gate
            if allow_2qubit:
                max_possible_operands = min(len(remaining_qubits), max_operands)
                num_operands = rng.choice(range(max_possible_operands)) + 1
            else:
                num_operands = 1

            # Selecting to which qubit(s) we apply the gate
            if num_operands == 1:
                rng.shuffle(remaining_qubits)
                operands = [remaining_qubits.pop() for _ in range(num_operands)]
                operation = rng.choice(one_q_ops)
            elif num_operands == 2:
                diff = 100
                while diff != 1:
                    rng.shuffle(remaining_qubits)
                    operands = remaining_qubits[-2:]
                    diff = np.abs(operands[0] - operands[1])
                operands = [remaining_qubits.pop() for _ in range(num_operands)]
                operation = rng.choice(two_q_ops)

            # Update quantum circuit
            register_operands = [qr[i] for i in operands]
            qc.append(operation(), register_operands)
        qc.barrier()
    return qc

if __name__ == "__main__":
    #disentangle_bell()
    disentangle_greedy_search()