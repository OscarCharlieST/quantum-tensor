import numpy as np
import matplotlib.pyplot as plt
import qtensor.states as states 
import qtensor.operators as ops
from qtensor.simulation.finiteTDVP import tdvp, right_mpo_contractions

# import os
# abspath = os.path.abspath(__name__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)

a = np.random.rand(2,3,3) + 1j * np.random.rand(2,3,3)
b = np.random.rand(2,3,3) + 1j * np.random.rand(2,3,3)

psi = states.mps([a, a, b, a, b])
psi.right_canonical()

H = ops.tilted_ising(N=5)

total_z_mpo = ops.total_z(5)
middle_z = ops.single_site_pauli(5, 2, 'z')
middle_x = ops.single_site_pauli(5, 2, 'x')

ops.mpo_expect(psi, H)

ops.mpo_expect(psi, total_z_mpo)

# state_hist, L_con, R_con = tdvp(psi, H, 1, 0.01, history=True)

ipsi = states.infinite_T_thermofield(5, 8, noise=0.01)
H_th = ops.thermofield_hamiltonian(H)
ipsi.right_canonical()
hist_th, l_th, r_th = tdvp(ipsi, H_th, 1, 0.01, history=True)
ipsi = hist_th[max(hist_th.keys())]
ipsi.right_canonical()
print(states.partite_entropy(ipsi, 2))

ent = states.partite_entropy(psi, 2)