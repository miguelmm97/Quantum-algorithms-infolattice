#%% Imports

# Built-in modules
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, HGate



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

#%% Circuit

# Create oracle
marked_states = ['100100']
oracle = grover_oracle(marked_states)

# Create Grover step
grover_op = GroverOperator(oracle, insert_barriers=True)

# Create full Grover's circuit
optimal_iter = math.floor(math.pi / (4 * math.asin(math.sqrt(len(marked_states) / 2 ** grover_op.num_qubits))))
qc = QuantumCircuit(grover_op.num_qubits)
qc.h(range(grover_op.num_qubits))
qc.compose(grover_op.power(optimal_iter), inplace=True)

# Measure all qubits
qc.measure_all()


#%% Figures
fig1 = oracle.draw(output="mpl", style="iqp")
ax1 = fig1.gca()
ax1.set_title('Oracle', fontsize=20)

fig2 = grover_op.decompose().draw(output="mpl", style="iqp")
ax2 = fig2.gca()
ax2.set_title('Grover iteration', fontsize=20)

fig3 = qc.draw(output="mpl", style="iqp")
ax3 = fig3.gca()
ax3.set_title('Full Grover circuit', fontsize=20)
plt.show()

