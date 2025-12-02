import numpy as np
from scipy.sparse import eye, kron, csr_matrix, hstack
from primme import eigsh
import info_lattice_Potts as il
import folded_info_lattice as fil
import save
import matplotlib.pyplot as plt
from parser import parse

import os

np.set_printoptions(threshold=np.inf)

if not os.path.exists("results_Potts"):
    os.makedirs("results_Potts")

# === 1. Define basic spin-1/2 basis states ===

up = np.array([1, 0])
down = np.array([0, 1])

def ket(*states):
    """ Tensor product of multiple states.
        Inputs:
            states: Vector of single-qubit states (numpy arrays).
        Output:
            result: Kronecker product state vector (csr_matrix).
        """

    result = states[0]
    for s in states[1:]:
        result = kron(result, s, format='csr')
    return result

# === 2. Define triplet basis in 2-qubit (4D) space ===

t0 = ket(up, up)
t1 = ((ket(up, down) + ket(down, up)) / np.sqrt(2))
t2 = ket(down, down)

# triplet_basis is a csr_matrix, which will be used to embed a qutrit operator (i.e., the original (3, 3) matrix acting
# on the qutrit) in the triplet sector of 2 qubits (i.e., making it a (4, 4) matrix)
triplet_basis = hstack([t0.T, t1.T, t2.T], format='csr')
triplet_basis.eliminate_zeros()

# === 3. Define qutrit (Potts) operators ===

d = 3  # qutrit dimension
omega = np.exp(2j * np.pi / d)
I_qutrit = np.eye(d, dtype=np.float64)  # qutrit identity

# Define shift operator X of the Potts model for qutrits and its Hermitian conjugate
X_op = csr_matrix(np.roll(I_qutrit, -1, axis=1))
X_op_dag = csr_matrix(X_op.T.conj())

# Define magnetization operator Z_op of the Potts model for qutrits and its Hermitian conjugate
Z_op = csr_matrix(np.diag([1, omega, omega**2]))
Z_op_dag = csr_matrix(Z_op.T.conj())

def delta_interaction_qutrit():
    """ Returns 9x9 matrix: delta(s_i, s_{i+1}) in qutrit ⊗ qutrit space.
        Inputs:
            None.
        Output:
            op: delta(s_i, s_{i+1}) (numpy array)
    """
    op = np.zeros((d*d, d*d))
    for k in range(d):
        proj = np.eye(1, d, k).T @ np.eye(1, d, k)
        op += np.kron(proj, proj)
    return op


# === 4. Embed qutrit operator into 4D space of the triplet sector of 2 qubits ===

# embed_one_site_qutrit_op() is used for the shift operator X and its Hermitian conjugate
def embed_one_site_qutrit_op(qutrit_op):
    """ Embed a 3x3 operator acting on 1 qutrit into a 4x4 operator on 2 qubits.
        Inputs:
            quitrit_op: numpy array.
        Output:
            op: delta(s_i, s_{i+1}) (numpy array).
        """
    return triplet_basis @ qutrit_op @ triplet_basis.T.conj()

# embed_two_site_qutrit_op() is used for the delta interaction
def embed_two_site_qutrit_op(qutrit_op):
    return kron(triplet_basis, triplet_basis) @ qutrit_op @ kron(triplet_basis, triplet_basis).conj().T


# === 5. Build full Hamiltonian ===

# Define a function to compute the kronecker product
def kron_n(ops):
    result = ops[0]
    for op in ops[1:]:
        result = kron(result, op, format='csr')
    return result

def build_embedded_two_site_operator_for_hamiltonian(two_site_operator, N, i):
    """Embed embedded 2-site operator acting on 2 qubits (triplet sector) at positions i and i+1 in N-site Hilbert space."""
    I = eye(4, format='csr')
    ops = []
    for site in range(N):
        if site == i:
            ops.append(csr_matrix(two_site_operator))
            skip_next = True
        elif site == i + 1:
            if 'skip_next' in locals():
                del skip_next
                continue
        else:
            ops.append(I)
    return kron_n(ops)

# Use the embedding functions to embed the Hamiltonian operators
# Onsite part
X_op_emb = csr_matrix(embed_one_site_qutrit_op(X_op))
X_op_dag_emb = csr_matrix(embed_one_site_qutrit_op(X_op_dag))
# Interaction part
delta_emb = embed_two_site_qutrit_op(delta_interaction_qutrit())

# Use the embedding functions to embed the Z_op operator (later use to compute order parameter)
Z_op_emb = csr_matrix(embed_one_site_qutrit_op(Z_op))

# Build the Hamiltonian as a csr_matrix
def build_full_hamiltonian(N, J, h):
    H_dim = 4**N
    H_zz = csr_matrix((H_dim, H_dim), dtype=np.float64)
    H_x = csr_matrix((H_dim, H_dim), dtype=np.float64)

    for i in range(N - 1):
        H_zz += build_embedded_two_site_operator_for_hamiltonian(delta_emb, N, i)
    for i in range(N):
        ops = [eye(4, format='csr')] * N
        ops[i] = X_op_emb + X_op_dag_emb
        H_x += kron_n(ops)

    H = -J * H_zz - h * H_x
    sparsity_H_zz = H_zz.count_nonzero() / H_dim**2
    sparsity_H_x = H_x.count_nonzero() / H_dim**2
    sparsity_H = H.count_nonzero() / H_dim**2
    return H


# === 6. Define functions to rotate the degenerate ground states in the SSB phase ===
# Note that the ground states obatined with some diagonalization routines could already respect the symmetry
# and the rotation is useless

def get_global_X_op(N, X_op_emb):
    # This is the product of X_i over all sites: prod_i X_i
    ops = [X_op_emb] * N
    return kron_n(ops)


def get_symmetry_sector_basis(eigvecs, num_states, N):
    print("Rotation to impose symmetry sector.")

    V = np.column_stack([eigvecs[:, i] for i in range(num_states)])
    X_op_global = get_global_X_op(N, X_op_emb)
    X_op_restricted = V.conj().T @ X_op_global @ V
    eigvals_sub, eigvecs_sub = np.linalg.eig(X_op_restricted)
    rotated_states = [V @ eigvecs_sub[:, i] for i in range(num_states)]
    return eigvals_sub, rotated_states


# === 7. Define functions to check for spontaneous symmetry breaking ===

# Define the functions to compute the expectation value of X_op
def get_local_X_op(N, site, X_op_emb):
    """Construct full operator for X acting on site `site`."""
    I4 = eye(4, format='csr')
    ops = [I4] * N
    ops[site] = X_op_emb
    return kron_n(ops)

def compute_order_parameter_X_op(psi, N):
    """
    psi       : dense state vector, shape (n,)
    N         : number of local operators
    X_op_emb : the embedded local operator (so get_local_X_op builds
                a sparse csr of shape (n,n) by inserting X_op_emb at site i)

    Return:
        The order parameter X, which is the product of X_i: prod_i X_i
    """

    # start with the vector we will transform
    phi = psi.copy()

    # apply each local X_op_i in turn:  phi <- X_op_i @ phi
    for i in range(N):
        Xi = get_local_X_op(N, i, X_op_emb)  # this is CSR
        phi = Xi.dot(phi)                    # sparse mat‐vec

    # now phi = (X_N ··· X_1) ψ, so the quadratic form is
    return np.vdot(psi, phi)

# Define the functions to compute the expectation value of Z_op
def get_local_Z_op(N, site, Z_op_emb):
    """Construct full operator for Z_op acting on site `site`."""
    I4 = eye(4, format='csr')
    ops = [I4] * N
    ops[site] = Z_op_emb
    return kron_n(ops)

def compute_order_parameter_Z(psi, N):
    """
    Return:
        The order parameter Z_op, which is the sum of Z_i: sum_i X_i
    """
    total = 0+0j
    for i in range(N):
        Zi_psi = get_local_Z_op(N, i, Z_op_emb).dot(psi)   # one sparse mat-vec
        total += np.vdot(psi, Zi_psi)                  # scalar
    return total / N


# === 8. Diagonalize Hamiltonian and compute ground state(s) ===

def run_exact_diagonalization(N, J, h, k, hdf5_filename='data.h5', save_data=False, show=False):

    # Build Hamiltonian
    H = build_full_hamiltonian(N, J, h)

    print("The Hamiltonian is built.")

    attributes = {}
    attributes['N'] = N
    attributes['J'] = J
    attributes['h'] = h
    attributes['k'] = k

    # Real, symmetry-related initial subspace (n x 3)
    X_op_global = get_global_X_op(N, X_op_emb)
    n = 4 ** N
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n,))
    V = np.column_stack([v, X_op_global @ v, X_op_global @ (X_op_global @ v)])
    Q, _ = np.linalg.qr(V)

    # Sparse diagonalization using PRIMME
    eigvals, eigvecs ,stats = eigsh(H, k=max(k,3), which='SA', return_stats=True, ncv=32, maxBlockSize=3, locking=False, tol=1e-12, method='PRIMME_JDQMR_ETol', v0=Q)  # GD+k
    print("Statistics of the sparse diagonalization with PRIMME", stats)
    rnorms = np.array(stats['rnorms'])

    # Delete H to save memory
    del H

    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    rnorms = rnorms[idx]

    print(f'Lowest {k} eigenvalues:', eigvals)

    # 1) Machine precision for dtype
    eps = float(np.finfo(eigvals.dtype).eps)  # ≃2e-16 for float64
    # 2) Energy scale in low-lying sector
    scale = max(abs(eigvals[0]),  abs(eigvals[1]), abs(eigvals[2]), 1.0)
    # 3) Solver accuracy scale: take the worst residual among the first 3
    resid_scale = np.max(rnorms[:3])
    # 4) Combine them
    tol = max(
        1e3 * eps * scale,  # “spectral” tolerance
        10 * resid_scale  # “solver” tolerance
    )

    degeneracy_flag = True

    gaps = np.abs(eigvals[1:3] - eigvals[0])

    print("gaps", gaps)
    print("tol", tol)

    if np.all(gaps < tol):
        print("Eigenvalues are degenerate and we are in the SSB phase.")
        _, low_energy_states = get_symmetry_sector_basis(eigvecs, d, N)

        # Compute the order parameter for each rotated eigenvector
        # and order them such that the one with expectation value ⟨Γ⟩=1 has index 0
        # In this way, I compute the information lattice always for the degenerate GS with the same ⟨Γ⟩

        X_op_expects = []
        for vec in low_energy_states:
            val = compute_order_parameter_X_op(vec, N)
            X_op_expects.append(val)

        idx = np.where(np.isclose(X_op_expects, 1))[0]
        if len(idx) == 0:
            print("WARNING: No eigenvector has X_op_expect = 1.")
        else:
            idx = idx[0]
            print(f"Eigenvector {idx} has X_op_expect = 1;\nWe consider this as the symmetric ground state for the info lattice.")

            # Reorder low_energy_states so that this eigenvector is at index 0
            if idx != 0:
                temp_eigvec = low_energy_states[0]
                low_energy_states[0] = low_energy_states[idx]
                low_energy_states[idx] = temp_eigvec
                del temp_eigvec

                temp_X_op_expect = X_op_expects[0]
                X_op_expects[0] = X_op_expects[idx]
                X_op_expects[idx] = temp_X_op_expect
                del temp_X_op_expect

    else:
        print("Eigenvalues are NOT degenerate.")
        degeneracy_flag = False
        low_energy_states = [eigvecs[:, i] for i in range(d)]

    attributes['degeneracy_flag'] = degeneracy_flag

    for i, psi in enumerate(low_energy_states):
        if save:
            save.save_low_energy_states(psi, i, hdf5_filename, attributes)
            save.save_low_energies(eigvals[i], i, hdf5_filename, attributes)

    del eigvecs

    # to improve the numerical efficiency we calculate only the information lattice and related quantities
    # only for the symmetric ground state
    print("Now we compute the information lattice and its derived quantities and other observables for the ground state.")

    X_op_expect = compute_order_parameter_X_op(low_energy_states[0], N)
    print(f"⟨X⟩ for low_energy_states[0]: {X_op_expect:.5f}")

    Z_expect = compute_order_parameter_Z(low_energy_states[0], N)
    print(f"⟨Z⟩ for low_energy_states[0]: {Z_expect:.5f}")

    info_latt = il.calc_info_latt(low_energy_states[0])

    info_per_scale = il.calc_info_per_scale(info_latt)
    print("info_per_scale", info_per_scale)

    tot_info_per_scale = np.sum(info_per_scale)
    print("tot_info_per_scale", tot_info_per_scale)

    Gamma = il.total_info_at_large_scales(info_per_scale)
    print(r"Gamma", Gamma)

    # Now compute the results for the folded information lattice
    info_latt_folded = fil.calc_info_latt_folded(low_energy_states[0])

    info_per_scale_folded = il.calc_info_per_scale(info_latt_folded)

    tot_info_per_scale_folded = np.sum(info_per_scale_folded)
    print("tot_info_per_scale_folded", tot_info_per_scale_folded)

    gamma_global = il.total_info_at_large_scales(info_per_scale_folded)
    print(r"gamma_global", gamma_global)


    if save_data:
        # save info lattice etc only for the symmetric ground state
        save.save_information_lattice(info_latt, 0, hdf5_filename, attributes)
        save.save_information_per_scale(info_per_scale, tot_info_per_scale, 0, hdf5_filename, attributes)
        save.save_Gamma(Gamma, 0, hdf5_filename, attributes)
        save.save_information_lattice_folded(info_latt_folded, 0, hdf5_filename, attributes)
        save.save_information_per_scale_folded(info_per_scale_folded, tot_info_per_scale, 0, hdf5_filename, attributes)
        save.save_gamma_global(gamma_global, 0, hdf5_filename, attributes)
        save.save_Hamiltonian_X_op_expect(X_op_expect, 0, hdf5_filename, attributes)
        save.save_Hamiltonian_Z_op_expect(Z_expect, 0, hdf5_filename, attributes)


    if show:
        if not os.path.exists("plots_Potts"):
            os.makedirs("plots_Potts")

        # plot the information lattice for the ground state
        il.plot_info_latt(info_latt, f"N={N}, J={J:.2f}, h={h:.2f}", f"./plots_Potts/info_latt_symmetric_ground_state_psi{0}_N{N}_J{J:.2f}_h{h:.2f}.pdf")

        # plot the information per scale
        plt.plot(info_per_scale, label='info per scale')
        plt.plot(info_per_scale_folded, label='info per scale folded')
        plt.title(rf"N={N}, J={J:.2f}, h={h:.2f}, $\Gamma$={Gamma:.3f}, $\gamma_\text{{global}}$={gamma_global:.3f}")
        plt.legend()
        plt.savefig(f"./plots_Potts/info_per_scale_symmetric_ground_state_psi{0}_N{N}_J{J:.2f}_h{h:.2f}.pdf")
        plt.show()

    return None


if __name__ == "__main__":
    args = parse()  # Get parameters from parser

    N = args.N # The number of spin-1 sites in the Potts model
    J = args.J # Interaction strength
    h = args.h_field # Transverse field strength
    k = args.k # The number of low-lying energy states to be computed with sparse diagonalization
    save_data = args.save_data
    show = args.show

    # File to save results
    hdf5_filename = f"./results_Potts/lowEnergyStates_N{N}_J{J:.3f}_h{h:.3f}_k{k}.h5"

    run_exact_diagonalization(N, J, h, k, hdf5_filename, save_data, show)
