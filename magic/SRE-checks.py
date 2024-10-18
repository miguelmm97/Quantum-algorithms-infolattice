# Built-in modules
import math
import numpy as np
from numpy import pi
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
from functions import stabiliser_Renyi_entropy_pure, stabiliser_Renyi_entropy_mixed


# Information lattice
from InfoLattice import calc_info, plot_info_latt, calc_info_per_scale
from functions import random_clifford_circuit


#%% Main

# Trivial state
num_qubits = 2
psi0_label = '0' * num_qubits
psi0 = Statevector.from_label(psi0_label)
SRE = stabiliser_Renyi_entropy_pure(psi0, 2, num_qubits)
print(f'Stabilizer Renyi entropy for |00>: {SRE}')

# Random clifford state
qc_clifford = random_clifford_circuit(num_qubits, 10)
SRE_cliff = stabiliser_Renyi_entropy_pure(psi0.evolve(qc_clifford), 2, num_qubits)
print(f'Stabilizer Renyi entropy for |00>: {SRE_cliff}')

# T gate state
qc_magic = QuantumCircuit(num_qubits)
qc_magic.h(range(num_qubits))
qc_magic.t(range(num_qubits))
qc_magic.cx(0, 1)
SRE_magic = stabiliser_Renyi_entropy_pure(psi0.evolve(qc_magic), 2, num_qubits)
th_value = - num_qubits * np.log(0.5 * (1 + np.cos(pi / 4) ** 4 + np.sin(pi / 4) ** 4))
print(f'Stabilizer Renyi entropy for HT|00>: {SRE_magic}, analytical value: {th_value}')

# Mixed states
psi = psi0.evolve(qc_magic)
rho = DensityMatrix(psi)
rho_A = partial_trace(rho, [0])
rho_B = partial_trace(rho, [1])
SRE_AB = stabiliser_Renyi_entropy_mixed(rho, num_qubits)
SRE_A = stabiliser_Renyi_entropy_mixed(rho_A, num_qubits - 1)
SRE_B = stabiliser_Renyi_entropy_mixed(rho_B, num_qubits - 1)
SRE_long = SRE_AB - SRE_A - SRE_B
print(f'Long range stabilizer Renyi entropy for HT|00>: {SRE_long}')


# qc_clifford.draw(output="mpl", style="iqp")
# qc_magic.draw(output="mpl", style="iqp")
plt.show()