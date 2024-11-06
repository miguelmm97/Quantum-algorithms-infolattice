#%% Imports

# Built-in modules
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Managing data
import os
import sys
import h5py
from datetime import date

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace

# Information lattice and functions
from modules.InfoLattice import calc_info
from modules.MagicLattice import calc_magic, calc_classical_magic
from modules.functions import *


#%% Logging setup
loger_main = logging.getLogger('main')
loger_main.setLevel(logging.INFO)

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
loger_main.addHandler(stream_handler)

#%% Parameters

# Initial state
n_qubits = 4
psi0_label = '0' * n_qubits
psi1 = Statevector.from_label(psi0_label)
psi2 = Statevector.from_label(psi0_label)

# Circuit parameters
Nlayers, Nblocks = 200, 20
T_per_block = 1
min_block, max_block = 0, 0
info_interval = int(Nlayers/ Nblocks)
seed_list = np.random.randint(0, 1000000, size=(Nlayers, ))
qubits = list(range(n_qubits))

# Preallocation
info_dict, info_dict_clifford = {}, {}
state_dict, state_dict_clifford  = {}, {}
magic, magic_dict = np.zeros((Nlayers, )), {}
SRE_clifford, SRE_long_clifford = np.zeros((Nlayers, )), np.zeros((Nlayers, ))
SRE, SRE_long = np.zeros((Nlayers, )), np.zeros((Nlayers, ))

#%% Circuit

clifford_sequence = QuantumCircuit(n_qubits)
magic_circuit = QuantumCircuit(n_qubits)
for i in range(Nlayers):

    # Generate Clifford layer
    layer = QuantumCircuit(n_qubits)
    clifford = random_clifford_circuit(num_qubits=n_qubits, depth=1, seed=seed_list[i])
    clifford_sequence.compose(clifford, inplace=True)
    layer.compose(clifford, inplace=True)

    # Application of T gates
    rng = np.random.default_rng(seed_list[i])
    rng.shuffle(qubits)
    operands = qubits[-T_per_block:]
    if (i % info_interval) == 0 and min_block < (i // info_interval) < max_block:
        for qubit in operands:
            layer.t(qubit)
    magic_circuit.compose(layer, inplace=True)

    # Information lattice and magic measures
    psi1 = psi1.evolve(layer)
    psi2 = psi2.evolve(clifford)
    info_latt = calc_info(psi1.data)
    info_latt_clifford = calc_info(psi2.data)
    # magic_latt = calc_magic(psi1.data)
    magic_latt = calc_classical_magic(psi1.data)


    magic[i] = non_integer_magic(info_latt)
    if (i % info_interval) == 0:
        info_dict[i // info_interval] = info_latt
        info_dict_clifford[i // info_interval] = info_latt_clifford
        magic_dict[i // info_interval] = magic_latt

#%% Saving data
data_dir = '/home/mfmm/Projects/quantum-algorithms-info/git-repo/magic/data'   # '../data' #
file_list = os.listdir(data_dir)
expID = get_fileID(file_list, common_name='Exp')
filename = '{}{}{}'.format('Exp', expID, '.h5')
filepath = os.path.join(data_dir, filename)

with h5py.File(filepath, 'w') as f:
    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation,      'seed',                seed_list)
    store_my_data(simulation,      'magic',               magic)
    store_my_data(simulation,      'SRE',                 SRE)
    store_my_data(simulation,      'SRE_long',            SRE_long)
    store_my_data(simulation,      'SRE_clifford',        SRE_clifford)
    store_my_data(simulation,      'SRE_long_clifford',   SRE_long_clifford)

    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters,       'n_qubits',           n_qubits)
    store_my_data(parameters,       'Nlayers',            Nlayers)
    store_my_data(parameters,       'Nblocks',            Nblocks)
    store_my_data(parameters,       'T_per_block',        T_per_block)
    store_my_data(parameters,       'min_block',          min_block)
    store_my_data(parameters,       'max_block',          max_block)

    # Attributes
    attr_my_data(parameters, "Date",       str(date.today()))
    attr_my_data(parameters, "Code_path",  sys.argv[0])

loger_main.info('Data saved correctly')

# Stabiliser Renyi entropies
# state_dict[i] = psi1.data
# state_dict_clifford[i] = psi2.data
# rho_clifford = DensityMatrix(psi2)
# rho_clifford_A = partial_trace(rho_clifford, [0])
# rho_clifford_B = partial_trace(rho_clifford, [1])
# SRE_clifford_AB = stabiliser_Renyi_entropy_mixed(rho_clifford, n_qubits)
# SRE_clifford_A = stabiliser_Renyi_entropy_mixed(rho_clifford_A, n_qubits - 1)
# SRE_clifford_B = stabiliser_Renyi_entropy_mixed(rho_clifford_B, n_qubits - 1)
# SRE_clifford[i] = stabiliser_Renyi_entropy_pure(psi2, 2, n_qubits)
# SRE_long_clifford[i] = SRE_clifford_AB - SRE_clifford_A - SRE_clifford_B

# rho = DensityMatrix(psi1)
# rho_A = partial_trace(rho, [0])
# rho_B = partial_trace(rho, [1])
# SRE_AB = stabiliser_Renyi_entropy_mixed(rho, n_qubits)
# SRE_A = stabiliser_Renyi_entropy_mixed(rho_A, n_qubits - 1)
# SRE_B = stabiliser_Renyi_entropy_mixed(rho_B, n_qubits - 1)
# SRE[i] = stabiliser_Renyi_entropy_pure(psi1, 2, n_qubits)
# SRE_long[i] = SRE_AB - SRE_A - SRE_B


# print(f'Layer: {i}, Info per scale |psi>:', calc_info_per_scale(info_dict[i], bc='open'))