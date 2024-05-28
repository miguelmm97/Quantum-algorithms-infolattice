#%% Imports

# Built-in modules
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, HGate
from qiskit.quantum_info import Statevector

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
marked_states = ['1000']
num_qubits = len(marked_states[0])
oracle_infolatt = grover_oracle(marked_states)
grover_op_infolatt = GroverOperator(oracle_infolatt, insert_barriers=True)

# Initial state
qc_infolatt = QuantumCircuit(num_qubits)
psi0 = Statevector.from_label('1000')
info0 = calc_info(psi0.data)

# First step: Hadamard transform
qc_infolatt.h(range(num_qubits))
psi1 = psi0.evolve(qc_infolatt)
info1 = calc_info(psi1.data)

# Second step: grover iteration 1
qc_infolatt.compose(grover_op_infolatt, inplace=True)
psi2 = psi0.evolve(qc_infolatt)
info2 = calc_info(psi2.data)

# Third step: grover iteration 2
qc_infolatt.compose(grover_op_infolatt, inplace=True)
psi3 = psi0.evolve(qc_infolatt)
info3 = calc_info(psi3.data)


#%% Figures
fig = plt.figure(figsize=(6, 8))
gs = GridSpec(2, 2, figure=fig, wspace=0.5, hspace=0.5)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

plot_info_latt(info0, ax1)
plot_info_latt(info0, ax2)
plot_info_latt(info0, ax3)
plot_info_latt(info0, ax4)
qc_infolatt.draw(output="mpl", style="iqp")
plt.show()














