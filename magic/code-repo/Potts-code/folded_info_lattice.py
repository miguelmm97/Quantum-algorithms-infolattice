import numpy as np
import info_lattice_Potts as il

"""=============================================================================

We compute the FOLDED information lattice and derived quantities
In this PBC-like folding we coarse grain the system by defining enlarged physical degrees of freedom
We pair as a new edge degree of freedom the initial two edges of the OBC chain,
the second last sites on the two initial edges as a new second last edge and so on.
The new other edge is the pair of the two sites initially residing in the center.

============================================================================="""


def reshape_psi_folded(psi, n, l):
    """
    Reshape the wavefunction for a folded chain with effective sites formed by pairing
    original open-boundary sites (i, L-1-i).

    Parameters
    ----------
    psi : array_like
        State vector of length 2**L.
    n : int
        Starting index of the block of length l in the folded chain (0 <= n <= N-l).
    l : int
        Number of effective sites in the block on the folded chain.

    Returns
    -------
    psi_mat : ndarray, shape (2**(2*l), 2**(L - 2*l))
        Reshaped matrix of the wavefunction for S_vN computation.
    """
    
    L = int(np.log2(psi.size))
    if L % 2 != 0:
        raise ValueError("System size must be even for folding.")

    # reshape into tensor of shape [2]*L
    psi = psi.reshape([2] * L)
    # collect axes for region A: pairs (k, L-1-k) for k in [n, n+l)
    axes_A = []
    for k in range(n, n + l):
        axes_A.extend([k, L - 1 - k])

    # remaining axes for complement
    axes_rest = [i for i in range(L) if i not in axes_A]
    # permute axes so that region A axes come first
    psi = np.transpose(psi, axes_A + axes_rest)
    # reshape to matrix
    return psi.reshape(2**(2*l), 2**(L - 2*l))


def calc_entropies_folded(psi):
    """
    Compute entanglement entropies S_vN for all contiguous blocks on the folded chain.

    Parameters
    ----------
    psi : array_like
        State vector of length 2**L (L even).

    Returns
    -------
    SvN_fold : dict
        SvN_fold[l] is a numpy array of entropies for all blocks of length l on the folded chain.
    """

    L = int(np.log2(psi.size))
    if L % 2 != 0:
        raise ValueError("System size must be even for folding.")
    N = L // 2
    SvN_folded = {}

    for l in range(1, N + 1):
        ent_list = []
        for n in range(N - l + 1):
            psi_r = reshape_psi_folded(psi, n, l)
            ent_list.append(il.S_vN(psi_r)) # Use the same function S_vN as for the unfolded case in info_lattice_Potts.py
        SvN_folded[l] = np.array(ent_list)
    return SvN_folded


def calc_info_latt_folded(psi):
    """
    Compute the information lattice on scale l for a PBC-like folded chain from an OBC state.

    Parameters
    ----------
    psi : array_like
        State vector of length 2**L (L even).

    Returns
    -------
    info_latt_fold : dict
        info_latt_fold[l] is the information on scale l after folding.
    """

    L = int(np.log2(psi.size))
    if L % 2 != 0:
        raise ValueError("System size must be even for folded info lattice.")
    N = L // 2
    SvN = calc_entropies_folded(psi)
    info_latt_fold = {}
    for l in range(1, N + 1):
        if l == 1:
            info_latt_fold[l] = 2 - SvN[l]
        elif l == 2:
            info_latt_fold[l] = - SvN[l] + SvN[l - 1][:-1] + SvN[l - 1][1:]
        else:
            info_latt_fold[l] = - SvN[l] + SvN[l - 1][:-1] + SvN[l - 1][1:] - SvN[l - 2][1:-1]
    return info_latt_fold










