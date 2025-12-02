import tables as tb
import numpy as np

"""=============================================================================

Script with the functions to save the information lattice, information per scale, eigenstates, eigenenergies, etc.

============================================================================="""


def save_information_lattice(information_lattice, d, file_path, attributes):
    group_name = f'/information_lattice'
    dset_name = f'info_latt_{d}'
    dset_path = f'{group_name}/{dset_name}'

    with tb.File(filename=file_path, mode='a') as h5f:
        if not dset_path in h5f:
            # Create a new table
            dtypes = [(f'n_{0}', 'f8')]
            for l in range(1, 2*attributes['N']):
                dtypes.append((f'n_{l}', 'f8'))

            h5table = h5f.create_table(
                where=group_name,  # Group name
                name=dset_name,  # Dataset name. Total path will be /some/group/trajectory
                title="Potts",
                description=np.dtype(dtypes),  # The dtype of each column on the table
                createparents=True,  # Create the intermediate groups if they do not exist yet
                chunkshape=None,  # Give the number of rows that should be treated as a chunk
                expectedrows=2*attributes['N'],  # Give the expected number of rows so that pytables can guess a chunkshape
                filters=tb.Filters(complevel=6,  # Compression level [0-9] (default is 6)
                                   complib="zlib",
                                   # Compression library (zlib is the default and most portable, but others may perform better)
                                   ),
            )

            # Store metadata attributes
            h5table.attrs.N = attributes['N']
            h5table.attrs.J = attributes['J']
            h5table.attrs.h = attributes['h']
            h5table.attrs.k = attributes['k']
            h5table.attrs.deg = attributes['degeneracy_flag']

        # Open the existing table to continue appending
        h5table = h5f.get_node(where=group_name, name=dset_name, classname="Table")

        data_values = [val for val in information_lattice[1]]
        h5table.append([tuple(data_values)])

        for ll in range(2, 2*attributes['N'] + 1):
            data_values = [val for val in information_lattice[ll]]
            data_values.extend(0. for ii in range(ll - 1))
            h5table.append([tuple(data_values)])


def save_information_per_scale(info_per_scale, tot_info_per_scale, d, file_path, attributes):
    group_name = f'/information_per_scale'
    dset_name = f'info_per_scale_{d}'
    dset_path = f'{group_name}/{dset_name}'

    with tb.File(filename=file_path, mode='a') as h5f:
        if not dset_path in h5f:
            # Create a new table
            dtypes = [(f'l_{0}', 'f8')]
            for l in range(1, 2*attributes['N']):
                dtypes.append((f'l_{l}', 'f8'))
            dtypes.append((f'tot', 'f8'))

            h5table = h5f.create_table(
                where=group_name,  # Group name
                name=dset_name,  # Dataset name. Total path will be /some/group/trajectory
                title="Potts",
                description=np.dtype(dtypes),  # The dtype of each column on the table
                createparents=True,  # Create the intermediate groups if they do not exist yet
                chunkshape=None,  # Give the number of rows that should be treated as a chunk
                expectedrows=2*attributes['N']+1,  # Give the expected number of rows so that pytables can guess a chunkshape
                filters=tb.Filters(complevel=6,  # Compression level [0-9] (default is 6)
                                   complib="zlib",
                                   # Compression library (zlib is the default and most portable, but others may perform better)
                                   ),
            )

            # Store metadata attributes
            h5table.attrs.N = attributes['N']
            h5table.attrs.J = attributes['J']
            h5table.attrs.h = attributes['h']
            h5table.attrs.k = attributes['k']
            h5table.attrs.deg = attributes['degeneracy_flag']

        # Open the existing table to continue appending
        h5table = h5f.get_node(where=group_name, name=dset_name, classname="Table")

        h5table.append([tuple(np.append(info_per_scale, tot_info_per_scale))])  # Append as a single row with L values


def save_information_lattice_folded(information_lattice, d, file_path, attributes):
    group_name = f'/information_lattice_folded'
    dset_name = f'info_latt_folded_{d}'
    dset_path = f'{group_name}/{dset_name}'

    with tb.File(filename=file_path, mode='a') as h5f:
        if not dset_path in h5f:
            # Create a new table
            dtypes = [(f'n_{0}', 'f8')]
            for l in range(1, attributes['N']):
                dtypes.append((f'n_{l}', 'f8'))

            h5table = h5f.create_table(
                where=group_name,  # Group name
                name=dset_name,  # Dataset name. Total path will be /some/group/trajectory
                title="Potts",
                description=np.dtype(dtypes),  # The dtype of each column on the table
                createparents=True,  # Create the intermediate groups if they do not exist yet
                chunkshape=None,  # Give the number of rows that should be treated as a chunk
                expectedrows=attributes['N'],  # Give the expected number of rows so that pytables can guess a chunkshape
                filters=tb.Filters(complevel=6,  # Compression level [0-9] (default is 6)
                                   complib="zlib",
                                   # Compression library (zlib is the default and most portable, but others may perform better)
                                   ),
            )

            # Store metadata attributes
            h5table.attrs.N = attributes['N']
            h5table.attrs.J = attributes['J']
            h5table.attrs.h = attributes['h']
            h5table.attrs.k = attributes['k']
            h5table.attrs.deg = attributes['degeneracy_flag']

        # Open the existing table to continue appending
        h5table = h5f.get_node(where=group_name, name=dset_name, classname="Table")

        data_values = [val for val in information_lattice[1]]
        h5table.append([tuple(data_values)])

        for ll in range(2, attributes['N'] + 1):
            data_values = [val for val in information_lattice[ll]]
            data_values.extend(0. for ii in range(ll - 1))
            h5table.append([tuple(data_values)])


def save_information_per_scale_folded(info_per_scale, tot_info_per_scale, d, file_path, attributes):
    group_name = f'/information_per_scale_folded'
    dset_name = f'info_per_scale_folded_{d}'
    dset_path = f'{group_name}/{dset_name}'

    with tb.File(filename=file_path, mode='a') as h5f:
        if not dset_path in h5f:
            # Create a new table
            dtypes = [(f'l_{0}', 'f8')]
            for l in range(1, attributes['N']):
                dtypes.append((f'l_{l}', 'f8'))
            dtypes.append((f'tot', 'f8'))

            h5table = h5f.create_table(
                where=group_name,  # Group name
                name=dset_name,  # Dataset name. Total path will be /some/group/trajectory
                title="Potts",
                description=np.dtype(dtypes),  # The dtype of each column on the table
                createparents=True,  # Create the intermediate groups if they do not exist yet
                chunkshape=None,  # Give the number of rows that should be treated as a chunk
                expectedrows=attributes['N']+1,  # Give the expected number of rows so that pytables can guess a chunkshape
                filters=tb.Filters(complevel=6,  # Compression level [0-9] (default is 6)
                                   complib="zlib",
                                   # Compression library (zlib is the default and most portable, but others may perform better)
                                   ),
            )

            # Store metadata attributes
            h5table.attrs.N = attributes['N']
            h5table.attrs.J = attributes['J']
            h5table.attrs.h = attributes['h']
            h5table.attrs.k = attributes['k']
            h5table.attrs.deg = attributes['degeneracy_flag']

        # Open the existing table to continue appending
        h5table = h5f.get_node(where=group_name, name=dset_name, classname="Table")

        h5table.append([tuple(np.append(info_per_scale, tot_info_per_scale))])  # Append as a single row with L values


def save_low_energy_states(psi, d, file_path, attributes):
    group_name = '/low_energy_states'
    array_name = f'low_energy_state_{d}'  # e.g. "psi0", "psi1", etc.

    dim = psi.shape[0]
    # Ensure dtype is complex128
    if psi.dtype != np.complex128:
        psi = psi.astype(np.complex128)

    with tb.File(filename=file_path, mode="a") as h5f:
        # 1) Create the group if missing
        if not h5f.__contains__(group_name):
            h5f.create_group('/', 'low_energy_states', createparents=True)

        full_path = f"{group_name}/{array_name}"
        # 2) If the array doesn't exist, create a CArray of shape (dim,) with Atom complex128
        if not h5f.__contains__(full_path):
            atom = tb.Atom.from_dtype(np.dtype('complex128'))
            carray = h5f.create_carray(
                where=group_name,
                name=array_name,
                atom=atom,
                shape=(dim,),
                title=f"Eigenvector {array_name}",
                createparents=True,
                filters=tb.Filters(complevel=5, complib="zlib")  # adjust compression if desired
            )
            # Store metadata attributes
            carray.attrs.N = attributes["N"]
            carray.attrs.J = attributes["J"]
            carray.attrs.h = attributes["h"]
            carray.attrs.k = attributes["k"]
            carray.attrs.deg = attributes["degeneracy_flag"]

        else:
            # If already exists, just open it
            carray = h5f.get_node(group_name, name=array_name, classname="CArray")

        # 3) Write the entire vector in one block
        carray[:] = psi

def save_low_energies(low_energy, d, file_path, attributes):
    group_name = '/low_energies'
    array_name = f'low_energy_{d}'

    with tb.File(filename=file_path, mode="a") as h5f:
        # 1) Ensure the group exists
        if group_name not in h5f:
            h5f.create_group('/', 'low_energies', createparents=True)

        # 2) If dataset exists, grab it; otherwise create a scalar Array
        try:
            arr = h5f.get_node(group_name, name=array_name)
            if not isinstance(arr, tb.Array):
                raise tb.NodeError(f"{group_name}/{array_name} is not a scalar Array")
        except tb.NoSuchNodeError:
            arr = h5f.create_array(
                where=group_name,
                name=array_name,
                obj=np.array(low_energy, dtype=np.float64),
                title=f"Low energy {array_name}",
                createparents=True
            )
            # 3) Store metadata
            arr.attrs.N = attributes["N"]
            arr.attrs.J = attributes["J"]
            arr.attrs.h = attributes["h"]
            arr.attrs.k = attributes["k"]
            arr.attrs.deg = attributes["degeneracy_flag"]

        # 4) Update the stored value
        arr[...] = low_energy

def save_Gamma(Gamma, d, file_path, attributes):
    group_name = '/Gammas'
    array_name = f'Gamma_{d}'

    with tb.File(filename=file_path, mode="a") as h5f:
        # 1) Ensure the group exists
        if group_name not in h5f:
            h5f.create_group('/', 'Gammas', createparents=True)

        # 2) If dataset exists, grab it; otherwise create a scalar Array
        try:
            arr = h5f.get_node(group_name, name=array_name)
            if not isinstance(arr, tb.Array):
                raise tb.NodeError(f"{group_name}/{array_name} is not a scalar Array")
        except tb.NoSuchNodeError:
            arr = h5f.create_array(
                where=group_name,
                name=array_name,
                obj=np.array(Gamma, dtype=np.float64),
                title=f"Gamma {array_name}",
                createparents=True
            )
            # 3) Store metadata
            arr.attrs.N = attributes["N"]
            arr.attrs.J = attributes["J"]
            arr.attrs.h = attributes["h"]
            arr.attrs.k = attributes["k"]
            arr.attrs.deg = attributes["degeneracy_flag"]

        # 4) Update the stored value
        arr[...] = Gamma


def save_gamma_global(gamma_global, d, file_path, attributes):
    group_name = '/gamma_global'
    array_name = f'gamma_global_{d}'

    with tb.File(filename=file_path, mode="a") as h5f:
        # 1) Ensure the group exists
        if group_name not in h5f:
            h5f.create_group('/', 'gamma_global', createparents=True)

        # 2) If dataset exists, grab it; otherwise create a scalar Array
        try:
            arr = h5f.get_node(group_name, name=array_name)
            if not isinstance(arr, tb.Array):
                raise tb.NodeError(f"{group_name}/{array_name} is not a scalar Array")
        except tb.NoSuchNodeError:
            arr = h5f.create_array(
                where=group_name,
                name=array_name,
                obj=np.array(gamma_global, dtype=np.float64),
                title=f"Gamma {array_name}",
                createparents=True
            )
            # 3) Store metadata
            arr.attrs.N = attributes["N"]
            arr.attrs.J = attributes["J"]
            arr.attrs.h = attributes["h"]
            arr.attrs.k = attributes["k"]
            arr.attrs.deg = attributes["degeneracy_flag"]

        # 4) Update the stored value
        arr[...] = gamma_global

def save_Hamiltonian_X_op_expect(X_op_expect, d, file_path, attributes):
    group_name = '/Hamiltonian_X_ops'
    array_name = f'Hamiltonian_X_op_{d}'

    with tb.File(filename=file_path, mode="a") as h5f:
        # 1) Ensure the group exists
        if group_name not in h5f:
            h5f.create_group('/', 'Hamiltonian_X_ops', createparents=True)

        # 2) If dataset exists, grab it; otherwise create a scalar Array
        try:
            arr = h5f.get_node(group_name, name=array_name)
            if not isinstance(arr, tb.Array):
                raise tb.NodeError(f"{group_name}/{array_name} is not a scalar Array")
        except tb.NoSuchNodeError:
            arr = h5f.create_array(
                where=group_name,
                name=array_name,
                obj=np.array(X_op_expect, dtype=np.complex128),
                title=f"Hamiltonian_X_op {array_name}",
                createparents=True
            )
            # 3) Store metadata
            arr.attrs.N = attributes["N"]
            arr.attrs.J = attributes["J"]
            arr.attrs.h = attributes["h"]
            arr.attrs.k = attributes["k"]
            arr.attrs.deg = attributes["degeneracy_flag"]

        # 4) Update the stored value
        arr[...] = X_op_expect

def save_Hamiltonian_Z_op_expect(Z_expect, d, file_path, attributes):
    group_name = '/Hamiltonian_Zs'
    array_name = f'Hamiltonian_Z_{d}'

    with tb.File(filename=file_path, mode="a") as h5f:
        # 1) Ensure the group exists
        if group_name not in h5f:
            h5f.create_group('/', 'Hamiltonian_Zs', createparents=True)

        # 2) If dataset exists, grab it; otherwise create a scalar Array
        try:
            arr = h5f.get_node(group_name, name=array_name)
            if not isinstance(arr, tb.Array):
                raise tb.NodeError(f"{group_name}/{array_name} is not a scalar Array")
        except tb.NoSuchNodeError:
            arr = h5f.create_array(
                where=group_name,
                name=array_name,
                obj=np.array(Z_expect, dtype=np.complex128),
                title=f"Hamiltonian_Z {array_name}",
                createparents=True
            )
            # 3) Store metadata
            arr.attrs.N = attributes["N"]
            arr.attrs.J = attributes["J"]
            arr.attrs.h = attributes["h"]
            arr.attrs.k = attributes["k"]
            arr.attrs.deg = attributes["degeneracy_flag"]

        # 4) Update the stored value
        arr[...] = Z_expect