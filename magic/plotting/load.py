import tables as tb
import numpy as np

# Loading functions corresponding to the save_* functions above

def load_information_lattice(d, file_path):
    """
    Load information_lattice data for a given index d.
    Returns:
        data: numpy array of shape (rows, 2*N)
        attrs: dict of metadata attributes (N, J, h, k)
    """
    group = '/information_lattice'
    dset = f'info_latt_{d}'

    with tb.File(file_path, mode='r') as h5f:
        try:
            table = h5f.get_node(group, dset)
        except tb.NoSuchNodeError:
            raise KeyError(f"Dataset {group}/{dset} not found in {file_path}")
        # Read full table
        data = table.read()
        # Extract metadata
        attrs = {key: getattr(table.attrs, key) for key in ['N', 'J', 'h', 'k']}
    return data, attrs


def load_information_per_scale(d, file_path):
    """
    Load information_per_scale data for a given index d.
    Returns:
        data: numpy array of shape (rows, 2*N+1)
        attrs: dict of metadata attributes (N, J, h, k)
    """
    group = '/information_per_scale'
    dset = f'info_per_scale_{d}'

    with tb.File(file_path, mode='r') as h5f:
        try:
            table = h5f.get_node(group, dset)
        except tb.NoSuchNodeError:
            raise KeyError(f"Dataset {group}/{dset} not found in {file_path}")
        data = table.read()
        attrs = {key: getattr(table.attrs, key) for key in ['N', 'J', 'h', 'k']}
    return data, attrs


def load_low_energy_states(d, file_path):
    """
    Load low energy state vector psi for index d.
    Returns:
        psi: numpy complex128 array
        attrs: dict of metadata attributes
    """
    group = '/low_energy_states'
    name = f'low_energy_state_{d}'

    with tb.File(file_path, mode='r') as h5f:
        try:
            carray = h5f.get_node(group, name)
        except tb.NoSuchNodeError:
            raise KeyError(f"CArray {group}/{name} not found in {file_path}")
        psi = carray.read()
        attrs = {key: getattr(carray.attrs, key) for key in ['N', 'J', 'h', 'k']}
    return psi, attrs


def load_low_energies(d, file_path):
    """
    Load low energy scalar for index d.
    Returns:
        energy: float
        attrs: dict of metadata
    """
    group = '/low_energies'
    name = f'low_energy_{d}'

    with tb.File(file_path, mode='r') as h5f:
        try:
            arr = h5f.get_node(group, name)
        except tb.NoSuchNodeError:
            raise KeyError(f"Array {group}/{name} not found in {file_path}")
        energy = arr.read()
        attrs = {key: getattr(arr.attrs, key) for key in ['N', 'J', 'h', 'k']}
    return energy, attrs


def load_Gamma(d, file_path):
    """
    Load Gamma scalar for index d.
    Returns:
        Gamma: float
        attrs: dict of metadata
    """
    group = '/Gammas'
    name = f'Gamma_{d}'

    with tb.File(file_path, mode='r') as h5f:
        try:
            arr = h5f.get_node(group, name)
        except tb.NoSuchNodeError:
            raise KeyError(f"Array {group}/{name} not found in {file_path}")
        Gamma = arr.read()
        attrs = {key: getattr(arr.attrs, key) for key in ['N', 'J', 'h', 'k']}
    return Gamma, attrs


def load_Lambda(d, file_path):
    """
    Load Lambda scalar for index d.
    Returns:
        Lambda: float
        attrs: dict of metadata
    """
    group = '/Lambdas'
    name = f'Lambda_{d}'

    with tb.File(file_path, mode='r') as h5f:
        try:
            arr = h5f.get_node(group, name)
        except tb.NoSuchNodeError:
            raise KeyError(f"Array {group}/{name} not found in {file_path}")
        Lambda = arr.read()
        attrs = {key: getattr(arr.attrs, key) for key in ['N', 'J', 'h', 'k']}
    return Lambda, attrs


def load_Hamiltonian_Gamma_expect(d, file_path):
    """
    Load Hamiltonian Gamma expectation (complex) for index d.
    Returns:
        value: complex
        attrs: dict of metadata
    """
    group = '/Hamiltonian_Gammas'
    name = f'Hamiltonian_Gamma_{d}'

    with tb.File(file_path, mode='r') as h5f:
        try:
            arr = h5f.get_node(group, name)
        except tb.NoSuchNodeError:
            raise KeyError(f"Array {group}/{name} not found in {file_path}")
        val = arr.read()
        attrs = {key: getattr(arr.attrs, key) for key in ['N', 'J', 'h', 'k']}
    return val, attrs


def load_Hamiltonian_Z_expect(d, file_path):
    """
    Load Hamiltonian Z expectation (complex) for index d.
    Returns:
        value: complex
        attrs: dict of metadata
    """
    group = '/Hamiltonian_Zs'
    name = f'Hamiltonian_Z_{d}'

    with tb.File(file_path, mode='r') as h5f:
        try:
            arr = h5f.get_node(group, name)
        except tb.NoSuchNodeError:
            raise KeyError(f"Array {group}/{name} not found in {file_path}")
        val = arr.read()
        attrs = {key: getattr(arr.attrs, key) for key in ['N', 'J', 'h', 'k']}
    return val, attrs


if __name__ == "__main__":
    # Example usage
    file_path = 'results_Potts/lowEnergyStates_N8_J1.000_h0.000.h5'
    d = 0
    info_latt, lat_attrs = load_information_lattice(d, file_path)
    print(f"Information lattice for d={d}: shape {info_latt.shape}, attrs={lat_attrs}")
    info_scale, scale_attrs = load_information_per_scale(d, file_path)
    print(f"Information per scale for d={d}: shape {info_scale.shape}, attrs={scale_attrs}")
    psi, psi_attrs = load_low_energy_states(d, file_path)
    print(f"Low energy state psi_{d}: length {psi.size}, attrs={psi_attrs}")
    e, e_attrs = load_low_energies(d, file_path)
    print(f"Low energy value for d={d}: {e}, attrs={e_attrs}")
    gamma, g_attrs = load_Gamma(d, file_path)
    print(f"Gamma_{d}: {gamma}, attrs={g_attrs}")
    lam, l_attrs = load_Lambda(d, file_path)
    print(f"Lambda_{d}: {lam}, attrs={l_attrs}")
    hG, hG_attrs = load_Hamiltonian_Gamma_expect(d, file_path)
    print(f"Hamiltonian Gamma expect for d={d}: {hG}, attrs={hG_attrs}")
    hZ, hZ_attrs = load_Hamiltonian_Z_expect(d, file_path)
    print(f"Hamiltonian Z expect for d={d}: {hZ}, attrs={hZ_attrs}")
