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

sigma_0 = np.eye(3, dtype=np.complex128)
sigma_x = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=np.complex128)
sigma_vec = [sigma_0, sigma_x, sigma_y, sigma_z]

#%% Functions



def spin(axis, site, L):
    prefactor = 1 if axis==3 else (1 / np.sqrt(2))
    return prefactor * np.kron(np.kron(np.eye(3 ** site), sigma_vec[axis]), np.eye(3 ** (L - site - 1)))


def Hamiltonian_XYZ(L, Jx, Jy, Jz, D):

    d = int(3 ** L)
    H = np.zeros((d, d), dtype=np.complex128)
    # H1 = np.zeros((d, d), dtype=np.complex128)
    # XYZ terms
    for i in range(L - 1):
        H += + Jx * spin(1, i, L) @ spin(1, i + 1, L) + \
             + Jy * spin(2, i, L) @ spin(2, i + 1, L) +\
             + Jz * spin(3, i, L) @ spin(3, i + 1, L)

    for i in range(L):
        H += D * spin(3, i, L) @ spin(3, i, L)

        # d_left = int(2 ** i)
        # d_right = int(2 ** (L - i - 2))
        # H1 += - Jx * (1/4) * np.kron(np.kron(np.kron(np.eye(d_left), sigma_x), sigma_x), np.eye(d_right))
        # H1 += - Jy * (1/4) * np.kron(np.kron(np.kron(np.eye(d_left), sigma_y), sigma_y), np.eye(d_right))
        # H1 += - Jz * (1/4) * np.kron(np.kron(np.kron(np.eye(d_left), sigma_z), sigma_z), np.eye(d_right))

    # Single ion anisotropy term
    # H += D * (1/4) * L * np.eye(d)

    T = np.eye(1)
    for i in range(L):
        T = np.kron(T, 1j * sigma_y)
    print(np.allclose(H, T @ np.conj(H) @ T.T.conj()))

    return H
