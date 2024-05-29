#%% Imports

# Built-in modules
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, HGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator

# Information lattice
from InfoLattice import calc_info, plot_info_latt


#%% Functions
def grover_oracle(marked_states):

    if not isinstance(marked_states, list):
        raise TypeError('marked_states should be a list of states in the computational basis.')

    # Initialise the circuit
    num_qubits = len(marked_states[0])
    qc = QuantumCircuit(num_qubits)

    # Mark each target state in the input list
    for target in marked_states:

        # Flip target bit-string to match Qiskit bit-ordering
        rev_target = target[::-1]

        # Create circuit that maps the target states
        zero_indx = [ind for ind in range(num_qubits) if rev_target.startswith('0', ind)]
        qc.x(zero_indx)
        qc.compose(MCMT(ZGate(), num_qubits - 1, 1), inplace=True)
        qc.x(zero_indx)

    return qc

#%% Evolving quantum states through the circuit

# Circuit details
marked_states = ['1010000']
num_qubits = len(marked_states[0])
oracle = grover_oracle(marked_states)
grover_op = GroverOperator(oracle, insert_barriers=True)
optimal_iter = math.floor(math.pi / (4 * math.asin(math.sqrt(len(marked_states) / 2 ** grover_op.num_qubits))))
info_dict = {}
n_iter = 8

# Initial state
qc = QuantumCircuit(num_qubits)
qc.h(range(num_qubits))
psi0 = Statevector.from_label('0000000')
info_dict[0] = calc_info(psi0.evolve(qc).data)

# Grover iterations
for i in range(1, n_iter + 1):
    qc.compose(grover_op, inplace=True)
    info_dict[i] = calc_info(psi0.evolve(qc).data)


#%% Figures
fig1 = plt.figure(figsize=(8, 5))
gs = GridSpec(2, 5, figure=fig1)

for i in range(n_iter + 1):
    if ((n_iter + 1) % 2) == 0:
        if i < int((n_iter + 1)/ 2):
            ax = fig1.add_subplot(gs[0, i])
        else:
            ax = fig1.add_subplot(gs[1, i % int((n_iter + 1)/ 2)])
    else:
        if i <= int((n_iter + 1) / 2):
            ax = fig1.add_subplot(gs[0, i])
        elif i != (n_iter + 1) - 1:
            ax = fig1.add_subplot(gs[1, (i % int((n_iter + 1) / 2)) - 1])
        else:
            ax = fig1.add_subplot(gs[1, int((n_iter + 1) / 2) - 1])

    plot_info_latt(info_dict[i], ax)
    ax.set_title(f'$G^{i}$')

fig1.suptitle(f'Optimal iteration: {optimal_iter}')
plt.show()






