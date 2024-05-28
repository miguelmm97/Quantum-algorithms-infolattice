#%% Imports

# Built-in modules
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Imports from Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, XGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator

# Information lattice
from InfoLattice import calc_info, plot_info_latt, calc_entropies

#%% Circuit creating Bell pairs

qc = QuantumCircuit(2)
qc.h(0)
qc.compose(MCMT(XGate(), 1, 1), inplace=True)
qc.draw(output="mpl", style="iqp")

psi0 = Statevector.from_label('10')
print(psi0.data.shape)
info0 = calc_info(psi0.data)

psi1 = psi0.evolve(qc)
info1 = calc_info(psi1.data)
SvN1 = calc_entropies(psi1.data)

fig1 = plt.figure(figsize=(6, 8))
gs = GridSpec(1, 2, figure=fig1, wspace=0.5, hspace=0.5)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[0, 1])

plot_info_latt(info0, ax1)
ax1.set_title('Step 0')
plot_info_latt(info1, ax2)
ax2.set_title('Step 1')
plt.show()

