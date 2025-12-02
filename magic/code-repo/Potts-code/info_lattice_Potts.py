import numpy as np
import matplotlib.pyplot as plt

"""=============================================================================

This scripts contains the functions to compute the information lattice and derived quantities
We assume we are dealing with a spin-1/2 chain of L sites with open boundary conditions

============================================================================="""

def reshape_psi(psi, n, l):
    """

    Parameters
    ----------
    psi : numpy array
        Vector of size 2**L where L is the number of sites.
    n : INT
        the leftmost site of the subset of interest A.
    l : INT
        The number of sites in the subset of interest A.

    Returns
    -------
    psi_r: numpy array (2**l,2**(L-l)) or (2**(L-l),2**l)
        Reshaped psi in a matrix psi_{(A),(\bar{A}) with (A) the combined
        indices in A, and similar for \bar{A}

    """

    L = int(np.log2(psi.size))
    psi = np.reshape(psi, L * [2])

    # If the block is smaller than its complement, bring it to the front
    if l <= L - l:
        axes = list(range(n, n + l)) + [i for i in range(L) if i < n or i >= n + l]
        # psi_r has shape (2^l, 2^(L-l))
        psi_r = np.transpose(psi, axes).reshape(2**l, -1)
    else:
        # Otherwise, use the complement block (entropy is equal)
        axes = [i for i in range(L) if not (n <= i < n + l)] + list(range(n, n + l))
        # psi_r has shape (2^(L-l), 2^l)
        psi_r = np.transpose(psi, axes).reshape(2**(L - l), -1)
        
    return psi_r


def S_vN(psi):
    """

    Parameters
    ----------
    psi : np.array 2**Na x 2**Nb
        The wave function. Assumed to have been reshpaed into a matrix with
        the correct subspace dimensions

    Returns
    -------
    S : FLOAT
    the von Neumann entanglement entropy computed using diagonalization of the density matrix rho

    """

    # Build the (small) Gram matrix: rho = A @ A^†
    rho = psi @ psi.conj().T

    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-16]
    return -(w * np.log2(w)).sum()


def calc_entropies(psi):
    """

    Parameters
    ----------
    psi : Numpy Array 2**L
        The wavefunction.

    Returns
    -------
    SvN : dictionary of np.arrays
        SvN[l] is the set of entropies of subspaces with size l

    """

    L = int(np.log2(psi.size))
    SvN = {}

    for l in range(1, L + 1):
        SvN[l] = []
        for n in range(L - l + 1):
            psi_r = reshape_psi(psi, n, l)
            SvN[l].append(S_vN(psi_r))
        SvN[l] = np.array(SvN[l])

    return SvN


def calc_info_latt(psi):
    """

    Parameters
    ----------
    psi : Numpy Array 2**L
        The wavefunction.

    Returns
    -------
    info_latt: dictionary of np.arrays
        info_latt[l] is the info on scale l
        Notice that here we use the convention that l goes from 1 to L

    """

    assert len(psi.shape) == 1
    L = int(np.log2(psi.size))
    SvN = calc_entropies(psi)
    info_latt = {}
    for l in range(1, L + 1):
        # At level 1 local information is just the von Neumann information of the single site
        if l == 1:
            info_latt[l] = l - SvN[l]

        # At level 2 local information is the von Neumann information of the two-site system minus that on the sigle sites
        elif l == 2:
            info_latt[l] = - SvN[l] + SvN[l - 1][:-1] + SvN[l - 1][1:]

        # On all larger levels local information is the the conditional mutual information
        else:
            info_latt[l] = -SvN[l] + SvN[l - 1][:-1] + SvN[l - 1][1:] - SvN[l - 2][1:-1]

    return info_latt


def calc_info_per_scale(info_latt):
    '''


    Parameters
    ----------
    info_latt : dictionary of numpy arrays
        info_latt[l] is the local information values on scale l

    Returns
    -------
    info_per_scale : numpy array
        info_per_scale[l] is the local information summed over all sites n on scale l

    '''

    L = len(info_latt)

    info_per_scale = np.zeros(L)

    for l in info_latt:
        info_per_scale[l - 1] = np.sum(info_latt[l])

    return info_per_scale


def diff_info_latt(info_latt_1, info_latt_2):
    """
    Compute the difference between two information lattice dictionaries

    Parameters
    ----------
    info_latt_1 : dict
        First information lattice dictionary
    info_latt_2 : dict
        Second information lattice dictionary

    Returns
    -------
    diff_latt : dict
        Dictionary containing the element-wise difference of the input dictionaries
    """

    diff_latt = {}

    # Get all the keys from the new information lattice
    all_keys_1 = set(info_latt_1.keys())
    all_keys_2 = set(info_latt_2.keys())

    assert all_keys_1==all_keys_2

    for key in all_keys_1:
        arr1 = info_latt_1.get(key)
        arr2 = info_latt_2.get(key)

        diff_latt[key] = arr1 - arr2

    return diff_latt


def total_info_at_large_scales(info_per_scale):
    """
   Compute Gamma, which is the sum of local information at large scales

   Parameters
   ----------
   info_per_scale : numpy array of length L

   Returns
   -------
   Gamma : scalar
       sum of of the information per scale on the largest L/2 scales
   """

    Gamma = np.sum(info_per_scale[int(np.floor(len(info_per_scale)/2)):])
    return Gamma


def plot_info_latt(info_latt, plot_title="title", plot_name="plot.pdf"):
    color_map = plt.get_cmap("Oranges")
    L = max(info_latt.keys())
    r = 1 / (4 * L)
    fig, ax = plt.subplots()
    for l in info_latt:
        for x in range(len(info_latt[l])):
            if np.abs(info_latt[l][x] - 0.) < 1e-2:
                ax.add_artist(plt.Circle((x / L + l / (2 * L), (l - 0.5) / L), r, color='white'))
            else:
                ax.add_artist(plt.Circle((x / L + l / (2 * L), (l - 0.5) / L), r, color=color_map(info_latt[l][x])))
    plt.xlim([-2 * r, 1])
    plt.ylim([-2 * r, 1 + 2 * r])
    ax.set_aspect('equal')
    plt.axis('off')
    plt.title(plot_title)
    plt.savefig(plot_name)
    plt.show()
