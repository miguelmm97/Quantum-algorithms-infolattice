#%% Imports

# Managing data
import os
import sys
import h5py
from datetime import date

# Imports from Qiskit
from qiskit import QuantumCircuit

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

# Information lattice and functions
from modules.functions import *
from modules.xyz_model import Hamiltonian_XYZ

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

Jx = 1
Jy = 1
Jz = 1
D = -0.1
L = 4

#%% Main

H = Hamiltonian_XYZ(L, Jx, Jy, Jz, D)
evals, evecs = np.linalg.eigh(H)

plt.plot(np.arange(len(evals)), evals, 'ob')
plt.show()

