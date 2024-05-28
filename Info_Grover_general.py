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
marked_states = ['1010000000']
num_qubits = len(marked_states[0])
oracle = grover_oracle(marked_states)
grover_op = GroverOperator(oracle, insert_barriers=True)
optimal_iter = math.floor(math.pi / (4 * math.asin(math.sqrt(len(marked_states) / 2 ** grover_op.num_qubits))))
info_dict = {}
n_iter = 8

# Initial state
qc = QuantumCircuit(num_qubits)
qc.h(range(num_qubits))
psi0 = Statevector.from_label('0000000000')

# Grover iterations
for i in range(0, n_iter):
    qc.compose(grover_op, inplace=True)
    info_dict[f'{i}'] = calc_info(psi0.evolve(qc).data)


#%% Figures
fig1 = plt.figure(figsize=(8, 5))
gs = GridSpec(2, 4, figure=fig1)
ax0 = fig1.add_subplot(gs[0, 0])
ax1 = fig1.add_subplot(gs[0, 1])
ax2 = fig1.add_subplot(gs[0, 2])
ax3 = fig1.add_subplot(gs[0, 3])
ax4 = fig1.add_subplot(gs[1, 0])
ax5 = fig1.add_subplot(gs[1, 1])
ax6 = fig1.add_subplot(gs[1, 2])
ax7 = fig1.add_subplot(gs[1, 3])
fig1.suptitle(f'Optimal iteration: {optimal_iter}')

plot_info_latt(info_dict['0'], ax0)
ax0.set_title(f'$G^1$')

plot_info_latt(info_dict['1'], ax1)
ax1.set_title(f'$G^2$')

plot_info_latt(info_dict['2'], ax2)
ax2.set_title(f'$G^3$')

plot_info_latt(info_dict['3'], ax3)
ax3.set_title(f'$G^4$')

plot_info_latt(info_dict['4'], ax4)
ax4.set_title(f'$G^5$')

plot_info_latt(info_dict['5'], ax5)
ax5.set_title(f'$G^6$')

plot_info_latt(info_dict['6'], ax6)
ax6.set_title(f'$G^7$')

plot_info_latt(info_dict['7'], ax7)
ax7.set_title(f'$G^8$')

plt.show()






