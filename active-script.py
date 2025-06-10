import numpy as np
import matplotlib.pyplot as plt
from qtensor.states import *
from qtensor.operators import *
from qtensor.simulation.finiteTDVP import *

# import os
# abspath = os.path.abspath(__name__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)

a = np.random.rand(2,3,3) + 1j * np.random.rand(2,3,3)
b = np.random.rand(2,3,3) + 1j * np.random.rand(2,3,3)

psi = mps([a, a, b, a, b])
psi.right_canonical()

H = tilted_ising(N=5)

total_z_mpo = total_z(5)
middle_z = single_site_pauli(5, 2, 'z')
middle_x = single_site_pauli(5, 2, 'x')

mpo_expect(psi, H)

R_con = right_mpo_contractions(psi, H)

mpo_expect(psi, total_z_mpo)
state_hist, L_con, R_con = tdvp(psi, H, 1, 0.01, history=True)