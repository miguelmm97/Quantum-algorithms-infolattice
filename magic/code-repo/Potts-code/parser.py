from argparse import ArgumentParser
from argparse import BooleanOptionalAction

def parse():
    parser = ArgumentParser(description='Potts')

    parser.add_argument('-N', type=int, help='Number of spins 1 in the chain', default=4)
    parser.add_argument('-J', type=float, help='Interaction strength', default=1)
    parser.add_argument('--h_field', type=float, help='Magnetic field strength', default=0.)
    parser.add_argument('-k', type=int, help='Number of eigenvalues and eigenvectors to compute', default=3)

    parser.add_argument('--save_data', help='If True save files with data', action=BooleanOptionalAction, default=True)
    parser.add_argument('--show', help='If True show plots', action=BooleanOptionalAction, default=True)

    args, unknown = parser.parse_known_args()

    return args



