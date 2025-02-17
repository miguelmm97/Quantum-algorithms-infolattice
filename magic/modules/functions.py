
# Math
import numpy as np

# Qiskit
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library.standard_gates import HGate, SGate, CXGate
from qiskit.circuit.exceptions import CircuitError

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

# Managing data
import os
import h5py


# %% Logging setup
def addLoggingLevel(levelName, levelNum, methodName=None):
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError("{} already defined in logging module".format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError("{} already defined in logging module".format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError("{} already defined in Logger class".format(methodName))

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


addLoggingLevel("TRACE", logging.DEBUG - 5)
logger_functions = logging.getLogger('functions')
logger_functions.setLevel(logging.INFO)

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
logger_functions.addHandler(stream_handler)

#%% Functions

# Circuits and magic measures
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

# Managing data
def get_fileID(file_list, common_name='datafile'):
    expID = 0
    for file in file_list:
        if file.startswith(common_name) and file.endswith('.h5'):
            stringID = file.split(common_name)[1].split('.h5')[0]
            ID = int(stringID)
            expID = max(ID, expID)
    return expID + 1

def store_my_data(file, name, data):
    try:
        file.create_dataset(name=name, data=data)
    except Exception as ex:
        logger_functions.warning(f'Failed to write {name} in {file} because of exception: {ex}')

def attr_my_data(dataset, attr_name, attr):
    try:
        dataset.attrs.create(name=attr_name, data=attr)
    except Exception as ex:
        logger_functions.warning(f'Failed to write {attr_name} in {dataset} because of exception: {ex}')

def load_my_data(file_list, directory):
    # Generate a dict with 1st key for filenames, 2nd key for datasets in the files
    data_dict = {}

    # Load desired directory and list files in it
    for file in file_list:
        file_path = os.path.join(directory, file)
        data_dict[file] = {}

        with h5py.File(file_path, 'r') as f:
            # Reading groups from the datafile
            for group in f.keys():
                # Reading subgroups/datasets from the group
                data_dict[file][group] = {}
                for subgroup in f[group].keys():
                    try:
                        # Reading datasets in the subgroup
                        data_dict[file][group][subgroup] = {}
                        for dataset in f[group][subgroup].keys():
                            if isinstance(f[group][subgroup][dataset][()], bytes):
                                data_dict[file][group][subgroup][dataset] = f[group][subgroup][dataset][()].decode()
                            else:
                                data_dict[file][group][subgroup][dataset] = f[group][subgroup][dataset][()]
                    except AttributeError:
                        try:
                            # Reading datasets from the group
                            if isinstance(f[group][subgroup][()], bytes):
                                data_dict[file][group][subgroup] = f[group][subgroup][()].decode()
                            else:
                                data_dict[file][group][subgroup] = f[group][subgroup][()]
                        # Case when there is no group
                        except AttributeError:
                            if isinstance(f[group][()], bytes):
                                data_dict[file][group] = f[group][()].decode()
                            else:
                                data_dict[file][group] = f[group][()]
    return data_dict

def load_my_attr(file_list, directory, dataset):
    attr_dict = {}

    # Load desired directory and list files in it
    for file in file_list:
        file_path = os.path.join(directory, file)
        attr_dict[file] = {}
        print(file)

        with h5py.File(file_path, 'r') as f:
            for att in f[dataset].attrs.keys():
                attr_dict[file][att] = f[dataset].attrs[att]

    return attr_dict

def store_my_dict(file, dict):
    for key in dict.keys():
        try:
            file.create_dataset(name=f'{key}', data=dict[key])
        except Exception as ex:
            logger_functions.warning(f'Failed to write key {key} in {file} because of exception: {ex}')

# Random
def kron_iter(L, up_position=0):
    """ Given a position returns |00...1...000...> in that position."""
    if L == 1:
        if up_position == L-1:
            return spinup()
        else:
            return spindown()
    else:
        if up_position == L-1:
            return np.kron(kron_iter(L - 1, up_position), spinup())
        else:
            return np.kron(kron_iter(L - 1, up_position), spindown())

def w_state_n(n):
    """ Creates a W-state of order n: |100...> + |010...> + ... """
    state = np.zeros(2**n, dtype=float)
    for i in range(n):
        state += kron_iter(n, i)
    state /= np.sqrt(n)
    return state

def wn_in_chain(L, n):
    """ The first n elements of this chain are W-like entangled."""
    assert L >= n
    if L == n:
        return w_state_n(n)
    else:
        return np.kron(wn_in_chain(L - 1, n), spindown())

def random_wn_state(n, k):
    order = np.random.default_rng().permutation(n)
    L = len(order)
    psi = wn_in_chain(L, k)
    psi = np.transpose(np.reshape(psi, [2] * L), axes=np.argsort(order))
    return psi.flatten()

def spinup():
    return np.array([1, 0])

def spindown():
    return np.array([0, 1])






