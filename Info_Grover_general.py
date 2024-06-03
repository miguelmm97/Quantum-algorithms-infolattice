#%% Imports

# Built-in modules
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, HGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator

# Information lattice
from InfoLattice import calc_info, plot_info_latt, calc_info_per_scale


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
marked_state_str = '100000'
marked_states = [marked_state_str]
num_qubits = len(marked_states[0])
oracle = grover_oracle(marked_states)
grover_op = GroverOperator(oracle, insert_barriers=True)
optimal_iter = math.floor(math.pi / (4 * math.asin(math.sqrt(len(marked_states) / 2 ** grover_op.num_qubits))))
info_dict = {}
n_iter = optimal_iter


# Probabilities for different iteration number
dim_H = 2 ** len(marked_state_str)
theta0 = 2 * np.arccos(np.sqrt((dim_H - len(marked_states))/ dim_H))
total_iter = 0.5 * (np.pi / theta0 - 1)
final_prob = np.sin((2 * np.ceil(total_iter) + 1) * theta0 / 2)

# Initial state
qc = QuantumCircuit(num_qubits)
qc.h(range(num_qubits))
phi0_str = '100001'
psi0 = Statevector.from_label(phi0_str)
info_dict[0] = calc_info(psi0.evolve(qc).data)

# Debug
if psi0.num_qubits != num_qubits:
    raise ValueError('Number of qubits of initial and marked states does not coincide.')

state = psi0.evolve(qc)
# Grover iterations
for i in range(1, n_iter + 1):
    print(f'iter: {i}')
    qc.compose(grover_op, inplace=True)
    info_dict[i] = calc_info(psi0.evolve(qc).data)
    state = psi0.evolve(qc)


#%% Information per scale
info_per_scale = {}
for step in info_dict.keys():
    info_per_scale[step] = calc_info_per_scale(info_dict[step], bc='open')
l_rescaled = np.arange(0, num_qubits) / (num_qubits  -1)

# Debug
for step in info_per_scale.keys():
    if not np.allclose(np.sum(info_per_scale[step]), num_qubits, atol=1e-15):
        raise ValueError(f'Information per scale does not add up! Error: {np.abs(num_qubits - np.sum(info_per_scale[step]))}')



#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
palette1 = seaborn.color_palette(palette='Blues', n_colors=n_iter+1)
palette2 = seaborn.color_palette(palette='dark:salmon_r', n_colors=n_iter+1)

# Information lattice per iteration
fig1 = plt.figure(figsize=(8, 5))
gs = GridSpec(2, int(np.ceil(n_iter / 2) + 1), figure=fig1)
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
    ax.set_title(f't: {i / optimal_iter:.2f}')
fig1.suptitle(f'Initial state: {phi0_str} , marked state: {marked_state_str},  optimal iteration: {optimal_iter}')

# Information per scale per iteration
fig2 = plt.figure(figsize=(8, 6.5))
ax2 = fig2.gca()
for step in info_dict.keys():
    ax2.plot(l_rescaled, info_per_scale[step], marker="o", label=f't = {step / optimal_iter:.2f}', color=palette1[step])
    ax2.plot(l_rescaled, info_per_scale[step], color=palette1[step])
ax2.set_xlim(0, 1)
ax2.set_ylim(1e-2, 10)
ax2.set_yscale('log')
ax2.set_xlabel("$l/l_{max}$", fontsize=20)
ax2.set_ylabel("$\log{(i_l)}$", fontsize=20)
ax2.tick_params(which='major', width=0.75, labelsize=15)
ax2.tick_params(which='major', length=10, labelsize=15)
ax2.legend(loc='best', ncol=2, fontsize=10, frameon=False)
ax2.set_title(f'Initial state: {phi0_str} , marked state: {marked_state_str},  optimal iteration: {optimal_iter}')



# fig3 = plt.figure(figsize=(20, 7.5))
# gs = GridSpec(2, 5, figure=fig3, wspace=0.5, hspace=0.5)
#
# for i in range(n_iter + 1):
#     if ((n_iter + 1) % 2) == 0:
#         if i < int((n_iter + 1)/ 2):
#             ax = fig3.add_subplot(gs[0, i])
#         else:
#             ax = fig3.add_subplot(gs[1, i % int((n_iter + 1)/ 2)])
#     else:
#         if i <= int((n_iter + 1) / 2):
#             ax = fig3.add_subplot(gs[0, i])
#         elif i != (n_iter + 1) - 1:
#             ax = fig3.add_subplot(gs[1, (i % int((n_iter + 1) / 2)) - 1])
#         else:
#             ax = fig3.add_subplot(gs[1, int((n_iter + 1) / 2) - 1])
#
#     ax.plot(l_rescaled, info_per_scale[i], marker="o", label=f't = {i / optimal_iter}', color=color_list[7])
#     ax.plot(l_rescaled, info_per_scale[i], alpha=0.3)
#     ax.set_xlim(0, 1)
#     ax.set_ylim(1e-2, 10)
#     ax.set_xlabel("$l/l_{max}$", fontsize=20)
#     ax.set_ylabel("$i_l$", fontsize=20)
#     ax.set_yscale('log')
#     ax.tick_params(which='major', width=0.75, labelsize=15)
#     ax.tick_params(which='major', length=10, labelsize=15)
#     ax.set_title(f'$G^{i}$', fontsize=20)


# fig3.suptitle(f'Optimal iteration: {optimal_iter}')


# fig4 = grover_op.decompose().draw(output="mpl", style="iqp")
# ax4 = fig4.gca()
# ax4.set_title('Grover iteration', fontsize=20)
plt.show()


