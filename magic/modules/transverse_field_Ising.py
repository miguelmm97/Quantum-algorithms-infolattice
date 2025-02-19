import numpy as np
from numpy import pi

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

#%% Logging setup
loger_xyz = logging.getLogger('XYZ')
loger_xyz.setLevel(logging.INFO)

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
loger_xyz.addHandler(stream_handler)


sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
sigma_vec = [sigma_0, sigma_x, sigma_y, sigma_z]


#%% Functions


def spin(axis, site, L):
    prefactor = 1
    return prefactor * np.kron(np.kron(np.eye(2 ** site), sigma_vec[axis]), np.eye(2 ** (L - site - 1)))



def Hamiltonian_transverse_field_Ising(theta, L, boundary='Periodic'):
    d = int(2 ** L)
    H = np.zeros((d, d), dtype=np.complex128)

    if boundary == 'Periodic':
        for i in range(L):
            H += -np.cos(theta) * spin(3, i, L) * spin(3, (i + 1) % L, L) - np.sin(theta) * spin(1, i, L)
    elif boundary == 'Open':
        for i in range(L - 1):
            H += - np.cos(theta) * spin(3, i, L) * spin(3, i + 1, L)
        for i in range(L):
            H += - np.sin(theta) * spin(1, i, L)

    return H