#%% Imports

# Built-in modules
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, HGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Clifford

# Information lattice
from InfoLattice import calc_info, plot_info_latt
from magic.modules.functions import random_clifford_circuit


#%% EPR pair state

qc = QuantumCircuit(2)
psi0_label = '0' * 2
psi0 = Statevector.from_label(psi0_label)

qc.h(0)
qc.cx(0, 1)
qc.t(0)
psi = psi0.evolve(qc)
info_latt = calc_info(psi.data)
# stab_group = Clifford(qc)
# print(stab_group)

fig1 = plt.figure()
gs = GridSpec(1, 1, figure=fig1)
ax1 = fig1.add_subplot(gs[0, 0])
plot_info_latt(info_latt, ax1)
qc.draw(output="mpl", style="iqp")
plt.show()


# clifford_undo = QuantumCircuit(2)
# clifford_undo.h(0)
# clifford_undo.cx(0, 1)
psi_try = psi1.evolve(clifford_sequence.inverse())
info_latt_try = calc_info(psi_try.data)
#
# print('Info per scale |psi>:', calc_info_per_scale(info_dict[i], bc='open'))
# print('Info per scale C|psi>:', calc_info_per_scale(info_latt_try, bc='open'))