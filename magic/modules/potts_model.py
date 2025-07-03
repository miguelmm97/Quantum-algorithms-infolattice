import numpy as np
from numpy import pi

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

#%% Logging setup
loger_potts = logging.getLogger('potts')
loger_potts.setLevel(logging.INFO)

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
loger_potts.addHandler(stream_handler)


X = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.complex128)
Z = np.diag([1, np.exp(2 * pi * 1j / 3), np.exp(4 * np.pi * 1j / 3)])

#%% Functions



def Z_operator(site, L):
    return np.kron(np.kron(np.eye(3 ** site), Z), np.eye(3 ** (L - site - 1)))

def X_operator(site, L):
    return np.kron(np.kron(np.eye(3 ** site), X), np.eye(3 ** (L - site - 1)))


def Hamiltonian_potts(L, J, g, boundary='Open'):

    d = int(3 ** L)
    H = np.zeros((d, d), dtype=np.complex128)

    if boundary == 'Open':
        for i in range(L - 1):
            H += - J * (Z_operator(i, L).T.conj() @ Z_operator(i+1, L) + Z_operator(i+1, L).T.conj() @ Z_operator(i, L))
        for i in range(L):
            H += - g * (X_operator(i, L) + X_operator(i, L).T.conj())

    return H
